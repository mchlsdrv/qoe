import datetime
import os
import pathlib

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
plt.style.use('ggplot')

VALIDATION_PROPORTION = 0.2
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/data/zoom/encrypted_traffic/test')
SAVE_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/data/zoom/encrypted_traffic/test/output')
os.makedirs(SAVE_DIR, exist_ok=True)


class QoEDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, features: list, labels: list):
        super().__init__()
        self.data = data
        data_cols = list(self.data.columns)

        # - Drop unused columns
        cols2drop = np.setdiff1d(data_cols, np.union1d(features, labels))
        self.data = self.data.drop(columns=cols2drop)

        # - Eliminate lines with at leas one NaN
        self.data = self.data.loc[self.data.isna().sum(axis=1) == 0]

        # Features
        self.features = self.data.loc[:, features].reset_index(drop=True)
        # - Normalize the features
        self.features_mu, self.features_std = self.features.mean(), self.features.std()
        self.features = (self.features - self.features_mu) / self.features_std

        # Labels
        self.labels = self.data.loc[:, labels].reset_index(drop=True)
        # - Normalize the labels
        self.labels_mu, self.labels_std = self.labels.mean(), self.labels.std()
        self.labels = (self.labels - self.labels_mu) / self.labels_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.as_tensor(self.features.iloc[index], dtype=torch.float32), torch.as_tensor(self.labels.iloc[index], dtype=torch.float32)

    def unnormalize_features(self, x):
        return x * self.features_std + self.features_mu

    def unnormalize_labels(self, y):
        return y * self.labels_std + self.labels_mu


class QoEModel(torch.nn.Module):
    def __init__(self, n_features, n_labels, n_layers, n_units):
        super().__init__()
        self.n_features = n_features
        self.n_labels = n_labels
        self.n_layers = n_layers
        self.n_units = n_units
        self.model = None
        self.layers = torch.nn.ModuleList()
        self._build()

    def _add_layer(self, n_in, n_out, activation):
        self.layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(n_in, n_out),
                torch.nn.BatchNorm1d(n_out),
                activation()
            )
        )

    def _build(self):
        # - Add the input layer
        self._add_layer(n_in=self.n_features, n_out=self.n_units, activation=torch.nn.SiLU)

        for lyr in range(self.n_layers):
            self._add_layer(n_in=self.n_units, n_out=self.n_units, activation=torch.nn.SiLU)

        self._add_layer(n_in=self.n_units, n_out=self.n_labels, activation=torch.nn.Tanh)

    def forward(self, x, p_drop: float = 0.0):
        tmp_in = x
        for lyr in self.layers:

            x = lyr(x)

            # - Skip connection
            if tmp_in.shape == x.shape:
                x = x + tmp_in

            tmp_in = x

        x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)

        return x


# class QoEModel(torch.nn.Module):
#     def __init__(self, n_features, n_labels, n_layers, n_units):
#         super().__init__()
#         self.n_features = n_features
#         self.n_labels = n_labels
#         self.n_layers = n_layers
#         self.n_units = n_units
#         self.model = None
#         self.layers = []
#         self._build()
#
#     def _build(self):
#         self.layers = [
#             torch.nn.Linear(self.n_features, self.n_units),
#             torch.nn.BatchNorm1d(self.n_units),
#             torch.nn.SiLU()
#         ]
#
#         for lyr in range(self.n_layers):
#             self.layers.append(torch.nn.Linear(self.n_units, self.n_units))
#             self.layers.append(torch.nn.BatchNorm1d(self.n_units))
#             self.layers.append(torch.nn.SiLU())
#
#         self.layers.append(torch.nn.Linear(self.n_units, self.n_labels))
#         self.layers.append(torch.nn.ReLU())
#
#         self.model = torch.nn.Sequential(*self.layers)
#
#     def forward(self, x):
#         return self.model(x)


def get_train_val_split(data: pd.DataFrame, validation_proportion: float = 0.2):
    n_data = len(data)
    data_indices = np.arange(n_data)

    n_val_items = int(n_data * validation_proportion)
    val_indices = np.random.choice(data_indices, n_val_items, replace=True)
    val_data = data.iloc[val_indices]

    train_indices = np.setdiff1d(data_indices, val_indices)
    train_data = data.iloc[train_indices]

    return train_data, val_data


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def run_train(model: torch.nn.Module, epochs: int, train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader,
              loss_function: torch.nn, optimizer: torch.optim, lr_reduce_frequency: int, lr_reduce_factor: float = 1.0, dropout_epoch_start: int = 20, p_dropout_init: float = 0.1, device: torch.device = torch.device('cpu')):
    train_losses = np.array([])
    val_losses = np.array([])
    for epch in range(epochs):
        p_drop = p_dropout_init * (epch // dropout_epoch_start)
        print(f'Epoch: {epch}/{epochs} ({100 * epch / epochs:.2f}% done)')
        print(f'\t ** INFO ** p_drop = {p_drop:.4f}')
        btch_train_losses = np.array([])
        btch_pbar = tqdm.tqdm(train_data_loader)
        for (X, Y) in btch_pbar:
            X = X.to(device)
            results = model(X, p_drop=p_drop)
            loss = loss_function(results, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            btch_train_losses = np.append(btch_train_losses, loss.item())

        # - Reduce learning rate every
        if epch > 0 and epch % lr_reduce_frequency == 0:
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = old_lr * lr_reduce_factor
            optimizer.param_groups[0]['lr'] = new_lr
            print(f'\t ** INFO ** The learning rate was changed from {old_lr} -> {new_lr}')

        train_losses = np.append(train_losses, btch_train_losses.mean())

        btch_val_losses = np.array([])
        with torch.no_grad():
            model.eval()
            for (X, Y) in val_data_loader:
                X = X.to(device)
                results = model(X)
                loss = loss_function(results, Y)
                btch_val_losses = np.append(btch_val_losses, loss.item())

            model.train()
        val_losses = np.append(val_losses, btch_val_losses.mean())

        print(f'''
        ===
        > Stats: train = {train_losses.mean():.4f}, val = {val_losses.mean():.4f}
        ===
        ''')

    return train_losses, val_losses


def run_test(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device = torch.device('cpu')):
    test_results = pd.DataFrame()
    for (X, Y) in data_loader:
        X = X.to(device)
        btch_preds = model(X)

        for y, pred in zip(Y, btch_preds):
            d = dict()

            # - Add labels
            for i, y_val in enumerate(y):
                d[f'true_{i}'] = np.float32(y_val.numpy())

            # - Add preds
            for i, pred_val in enumerate(pred):
                d[f'pred_{i}'] = np.float32(pred_val.detach().numpy())

            # - Create the cv_5_folds frame
            btch_results = pd.DataFrame(d, index=pd.Index([0]))

            # - Add the batch cv_5_folds frame to the total results
            test_results = pd.concat([test_results, btch_results])

    test_results = test_results.reset_index(drop=True)

    return test_results


def unnormalize_results(results: pd.DataFrame, data_set: QoEDataset, n_columns: int) -> pd.DataFrame:
    """
    This function unnormalizes the labels by performing X * STD(X) + MEAN(X) performed in the process of dataset creation, thus it requires
    the original QoEDataset object
    :param results: pandas.DataFrame object containing the results with  2 * n_columns columns, where the first n_columns are the true, and the last n_columns are the predicted labels
    :param data_set: The original QoEDataset object which was created for the process of training / testing, and contains the mean and the std of each label
    :param n_columns: The number of labels
    :return: pandas.DataFrame containing the unnormalized true and predicted labels
    """
    unnormalized_results = pd.DataFrame()
    for line_idx in range(len(results)):
        # - Get the line
        res = pd.DataFrame(results.iloc[line_idx]).T.reset_index(drop=True)

        # - Get the true labels
        labels = pd.DataFrame(data_set.unnormalize_labels(res.iloc[0, :n_columns].values)).T

        # - Rename the columns to include the "true" postfix
        for old_name in data_set.data.columns:
            new_name = f'{old_name}_true'
            labels = labels.rename(columns={old_name: new_name})

        # - Get the predictions
        preds = pd.DataFrame(data_set.unnormalize_labels(res.iloc[0, n_columns:].values)).T
        for old_name in data_set.data.columns:
            new_name = f'{old_name}_pred'
            preds = preds.rename(columns={old_name: new_name})

        # - Concatenate the labels with the preds horizontally
        labels_preds = pd.concat([labels, preds], axis=1)

        # - Append to the unnormalized_results
        unnormalized_results = pd.concat([unnormalized_results, labels_preds])

    # - Reset the index to normal
    unnormalized_results = unnormalized_results.reset_index(drop=True)

    return unnormalized_results


def get_number_of_parameters(model: torch.nn.Module):

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_non_trainable_parameters = n_total_params - n_trainable_params

    return n_trainable_params, n_non_trainable_parameters


def get_errors(results: pd.DataFrame, columns: list):
    n_columns = len(columns)
    true = results.iloc[:, :n_columns].values
    pred = results.iloc[:, n_columns:].values

    columns = [column_name + '_errors(%)' for column_name in columns]

    errors = pd.DataFrame(100 - true * 100 / pred, columns=columns)

    return errors


def run_ablation(test_data_root: pathlib.Path, features: list, labels: list, batch_size_numbers: list, epoch_numbers: list, layer_numbers: list, unit_numbers: list, loss_functions: list, optimizers: list, initial_learning_rates: list,
                 save_dir: pathlib.Path):

    # - Will hold the final results
    ablation_results = pd.DataFrame()

    # - Will hold the metadata for all experiments
    test_metadata = pd.DataFrame()
    test_dir_names = os.listdir(test_data_root)

    # - Total experiments
    cv_folds = len(test_dir_names)
    n_batch_sizes = len(batch_size_numbers)
    n_epoch_numbers = len(epoch_numbers)
    n_layer_numbers = len(layer_numbers)
    n_unit_numbers = len(unit_numbers)
    n_loss_funcs = len(loss_functions)
    n_optimizers = len(optimizers)
    n_init_lrs = len(initial_learning_rates)
    total_exp = cv_folds * n_batch_sizes * n_epoch_numbers * n_layer_numbers * n_unit_numbers * n_loss_funcs * n_optimizers * n_init_lrs
    print(f'''
    =================================================================================================================================================================================
    = ***************************************************************************************************************************************************************************** =
    =================================================================================================================================================================================
            > Total Number of experiments in ablation:
                - CV folds: {cv_folds} 
                - Batch sizes: {n_batch_sizes}
                - Epoch numbers: {n_epoch_numbers}
                - Layer numbers: {n_layer_numbers}
                - Unit numbers: {n_unit_numbers}
                - Loss functions: {n_loss_funcs}
                - Optimizers: {n_optimizers}
                - Initial learning rates: {n_init_lrs}
                => N = {cv_folds} x {n_batch_sizes} x {n_epoch_numbers} x {n_layer_numbers} x {n_unit_numbers} x {n_loss_funcs} x {n_optimizers} x {n_init_lrs} = {total_exp}
    =================================================================================================================================================================================
    = ***************************************************************************************************************************************************************************** =
    =================================================================================================================================================================================
            ''')

    exp_idx = 0
    for fldr_idx, test_dir_name in enumerate(test_dir_names):

        # - Get the path to the current cv_5_folds folder
        data_folder = test_data_root / test_dir_name

        print(f'\t - Data folders: {data_folder} - {fldr_idx + 1}/{cv_folds} ({100 * (fldr_idx + 1) / cv_folds:.2f}% done)')

        # - Get the train / test cv_5_folds
        train_data_df = pd.read_csv(data_folder / 'train_data.csv')
        test_data_df = pd.read_csv(data_folder / 'test_data.csv')

        for epch_idx, n_epochs in enumerate(epoch_numbers):
            # print('******************************************************************************************')
            # print(f'\t - Epochs: {n_epochs} - {epch_idx + 1}/{len(epoch_numbers)} ({100 * (epch_idx + 1) / len(epoch_numbers):.2f}% done)')
            # print('******************************************************************************************')
            for btch_idx, batch_size in enumerate(batch_size_numbers):
                # print('******************************************************************************************')
                # print(f'\t - Batch size: {batch_size} - {btch_idx + 1}/{len(batch_size_numbers)} ({100 * (btch_idx + 1) / len(batch_size_numbers):.2f}% done)')
                # print('******************************************************************************************')
                for lyr_idx, n_layers in enumerate(layer_numbers):
                    # print('******************************************************************************************')
                    # print(f'\t - Layers number: {n_layers} - {lyr_idx + 1}/{len(layer_numbers)} ({100 * (lyr_idx + 1) / len(layer_numbers):.2f}% done)')
                    # print('******************************************************************************************')
                    for units_idx, n_units in enumerate(unit_numbers):
                        # print('******************************************************************************************')
                        # print(f'\t - Units number: {n_units} - {units_idx + 1}/{len(unit_numbers)} ({100 * (units_idx + 1) / len(unit_numbers):.2f}% done)')
                        # print('******************************************************************************************')
                        for loss_func_idx, loss_func in enumerate(loss_functions):
                            loss_func_name = 'mse'
                            # print('******************************************************************************************')
                            # print(f'\t - Loss function: {loss_func} - {loss_func_idx + 1}/{len(loss_functions)} ({100 * (loss_func_idx + 1) / len(loss_functions):.2f}% done)')
                            # print('******************************************************************************************')
                            for opt_idx, opt in enumerate(optimizers):
                                opt_name = 'adam'
                                if opt == torch.optim.SGD:
                                    opt_name = 'sgd'
                                if opt == torch.optim.Adamax:
                                    opt_name = 'adamax'
                                # print('******************************************************************************************')
                                # print(f'\t - Optimizer: {opt} - {opt_idx + 1}/{len(optimizers)} ({100 * (opt_idx + 1) / len(optimizers):.2f}% done)')
                                # print('******************************************************************************************')
                                for init_lr_idx, init_lr in enumerate(initial_learning_rates):
                                    # print('******************************************************************************************')
                                    # print(f'\t - Initial learning rate: {init_lr} - {init_lr_idx + 1}/{len(initial_learning_rates)} ({100 * (init_lr_idx + 1) / len(initial_learning_rates):.2f}% done)')
                                    # print('******************************************************************************************')

                                    exp_idx += 1

                                    # - Split into train / val
                                    train_df, val_df = get_train_val_split(train_data_df, validation_proportion=VALIDATION_PROPORTION)

                                    # - Dataset
                                    train_ds, val_ds = QoEDataset(data=train_df, features=features, labels=labels), QoEDataset(data=val_df, features=features, labels=labels)

                                    # - Data Loader
                                    train_dl = torch.utils.data.DataLoader(
                                        train_ds,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=1,
                                        drop_last=True
                                    )

                                    val_batch_size = batch_size // 4
                                    val_dl = torch.utils.data.DataLoader(
                                        val_ds,
                                        batch_size=val_batch_size if val_batch_size > 0 else 1,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=1,
                                        drop_last=True
                                    )

                                    # - Build the model
                                    mdl = QoEModel(n_features=len(features), n_labels=len(labels), n_layers=n_layers, n_units=n_units)
                                    n_train_params, n_non_train_params = get_number_of_parameters(model=mdl)
                                    mdl.to(DEVICE)

                                    # - Train
                                    # - Create the train directory
                                    train_save_dir = save_dir / f'Data={test_dir_name}_Epochs={n_epochs}_Batch={batch_size}_Layers={n_layers}_Units={n_units}_Opt={opt_name}_lr={init_lr}'
                                    os.makedirs(train_save_dir, exist_ok=True)

                                    # - Train the model
                                    train_losses, val_losses = run_train(
                                        model=mdl,
                                        epochs=n_epochs,
                                        train_data_loader=train_dl,
                                        val_data_loader=val_dl,
                                        loss_function=loss_func(),
                                        optimizer=opt(mdl.parameters(), lr=init_lr),
                                        lr_reduce_frequency=LR_REDUCE_FREQUENCY,
                                        lr_reduce_factor=LR_REDUCE_FACTOR,
                                        dropout_epoch_start=DROPOUT_EPOCH_START,
                                        p_dropout_init=P_DROPOUT_INIT,
                                        device=DEVICE
                                    )

                                    # - Save the train / val loss metadata
                                    np.save(train_save_dir / 'train_losses.npy', train_losses)
                                    np.save(train_save_dir / 'val_losses.npy', val_losses)

                                    # - Plot the train / val losses
                                    plt.plot(train_losses, label='train')
                                    plt.plot(val_losses, label='val')
                                    plt.suptitle('Train / Validation Loss Plot')
                                    plt.legend()
                                    plt.savefig(train_save_dir / 'train_val_loss.png')
                                    plt.close()

                                    # - Get the test dataloader
                                    test_ds = QoEDataset(data=test_data_df, features=features, labels=labels)
                                    test_dl = torch.utils.data.DataLoader(
                                        test_ds,
                                        batch_size=16,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=1,
                                        drop_last=True
                                    )

                                    # - Test the model
                                    test_res = run_test(model=mdl, data_loader=test_dl, device=DEVICE)
                                    test_res = unnormalize_results(results=test_res, data_set=test_ds, n_columns=len(test_res.columns) // 2)

                                    # - Save the test metadata
                                    test_res.to_csv(train_save_dir / f'test_results.csv', index=False)

                                    # - Get the test errors
                                    test_errs = get_errors(results=test_res, columns=test_ds.labels.columns)

                                    # - Save the test metadata
                                    test_errs.to_csv(train_save_dir / f'test_errors.csv', index=False)

                                    test_res = pd.concat([test_res, test_errs], axis=1)

                                    # configuration_test_metadata = pd.concat([test_metadata, test_res], axis=1).reset_index(drop=True)

                                    test_res.loc[:, 'dataset'] = test_dir_name
                                    test_res.loc[:, 'epochs'] = n_epochs
                                    test_res.loc[:, 'batch_size'] = batch_size
                                    test_res.loc[:, 'layers'] = n_layers
                                    test_res.loc[:, 'units'] = n_units
                                    test_res.loc[:, 'loss'] = loss_func_name
                                    test_res.loc[:, 'optimizer'] = opt_name
                                    test_res.loc[:, 'initial_lr'] = init_lr
                                    test_res.loc[:, 'n_params'] = n_train_params + n_non_train_params
                                    test_res.loc[:, 'n_train_params'] = n_train_params
                                    test_res.loc[:, 'n_non_train_params'] = n_non_train_params

                                    test_res.to_csv(train_save_dir / 'test_metadata.csv', index=False)

                                    # - Add the configuration test metadata to the global test metadata
                                    test_metadata = pd.concat([test_metadata, test_res]).reset_index(drop=True)

                                    # - Save the final metadata and the results
                                    test_metadata.to_csv(save_dir / 'test_metadata_tmp.csv', index=False)

                                    # - Get the results for the current configuration
                                    configuration_results = pd.DataFrame(
                                        dict(
                                            dataset=test_dir_name,
                                            epochs=n_epochs,
                                            batch_size=batch_size,
                                            layers=n_layers,
                                            units=n_units,
                                            loss=loss_func_name,
                                            optimizer=opt_name,
                                            initial_lr=init_lr,
                                            n_params=n_train_params + n_non_train_params,
                                            n_train_params=n_train_params,
                                            n_non_train_params=n_non_train_params
                                        ),
                                        index=pd.Index([0])
                                    )

                                    # -- Add the errors to the configuration results
                                    configuration_results = pd.concat([configuration_results, pd.DataFrame(test_errs.mean()).T], axis=1)

                                    # - Add the results for the current configuration to the final ablation results
                                    ablation_results = pd.concat([ablation_results, configuration_results], axis=0).reset_index(drop=True)

                                    # - Save the final metadata and the results
                                    ablation_results.to_csv(save_dir / 'ablation_final_results_tmp.csv', index=False)

                                    print(f'''
===========================================================
=================== Final Stats ===========================
===========================================================
Configuration:
    > {n_epochs} epochs
    > {batch_size} batch size
    > {n_layers} layers
    > {n_units} units
    > {n_epochs} epochs

Number of Parameters:
    > Trainable: {n_train_params}
    > Non-Trainable: {n_non_train_params}
    => Total: {n_train_params + n_non_train_params}

Mean Errors:   
{test_errs.mean()}

Status:
    - Experiment {exp_idx}/{total_exp} ({100 * exp_idx / total_exp:.2f}% done)
===========================================================
                                    ''')

    # - Save the final metadata and the results
    ablation_results.to_csv(save_dir / 'ablation_final_results.csv', index=False)

    # - Save the final metadata and the results
    test_metadata.to_csv(save_dir / 'test_metadata.csv', index=False)


N_LAYERS = 64
N_UNITS = 64
EPOCHS = 100
OPTIMIZER = torch.optim.Adam
INIT_LR = 0.01
# INIT_LR = 0.0015
LR_REDUCE_FREQUENCY = 20
LR_REDUCE_FACTOR = 0.5
# LR = 0.001
DROPOUT_EPOCH_START = 20
P_DROPOUT_INIT = 0.1
LOSS_FUNCTION = torch.nn.MSELoss()
TRAIN_BATCH_SIZE = 16

if __name__ == '__main__':
    # - Get the cv_5_folds
    train_data_df = pd.read_csv(DATA_ROOT / 'train_data.csv')

    train_df, val_df = get_train_val_split(train_data_df, validation_proportion=0.2)

    features = ['Bandwidth', 'pps', 'avg time between packets', 'packets length']
    # features = ['Bandwidth', 'pps', 'avg time between packets', 'packets length', 'Resolution', 'fps', 'Latency', 'Jitter']
    # labels = ['NIQE']
    labels = ['NIQE', 'Resolution', 'fps']

    # - Dataset
    train_ds, val_ds = QoEDataset(data=train_df, features=features, labels=labels), QoEDataset(data=val_df, features=features, labels=labels)

    # - Data Loader
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        drop_last=True
    )

    VAL_BATCH_SIZE = TRAIN_BATCH_SIZE // 4
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE if VAL_BATCH_SIZE > 0 else 1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=True
    )

    # - Build the model
    mdl = QoEModel(n_features=len(features), n_labels=len(labels), n_layers=N_LAYERS, n_units=N_UNITS)
    mdl.to(DEVICE)

    # - Optimizer
    optimizer = OPTIMIZER(mdl.parameters(), lr=INIT_LR)

    # - Loss
    loss_func = LOSS_FUNCTION

    # - Train
    # - Create the train directory
    train_save_dir = SAVE_DIR / f'{N_LAYERS}_layers_{N_UNITS}_units_{EPOCHS}_epochs_{TS}'
    os.makedirs(train_save_dir)
    train_losses, val_losses = run_train(
        model=mdl,
        epochs=EPOCHS,
        train_data_loader=train_dl,
        val_data_loader=val_dl,
        loss_function=loss_func,
        optimizer=optimizer,
        lr_reduce_frequency=LR_REDUCE_FREQUENCY,
        lr_reduce_factor=LR_REDUCE_FACTOR,
        dropout_epoch_start=DROPOUT_EPOCH_START,
        p_dropout_init=P_DROPOUT_INIT,
        device=DEVICE
    )

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.suptitle('Train / Validation Loss Plot')
    plt.legend()
    plt.savefig(train_save_dir / 'train_val_loss.png')
    plt.close()

    test_data_df = pd.read_csv(DATA_ROOT / 'test_data.csv')
    test_ds = QoEDataset(data=test_data_df, features=features, labels=labels)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=True
    )

    test_res = run_test(model=mdl, data_loader=test_dl, device=DEVICE)
    test_res = unnormalize_results(results=test_res, data_set=test_ds, n_columns=len(test_res.columns) // 2)
    test_res.to_csv(train_save_dir / f'test_results_{N_LAYERS}_layers_{N_UNITS}_units_{EPOCHS}_epochs.csv')

    test_errs = get_errors(results=test_res, columns=test_ds.labels.columns)
    test_errs.to_csv(train_save_dir / f'test_errors_{N_LAYERS}_layers_{N_UNITS}_units_{EPOCHS}_epochs.csv')

    test_res = pd.concat([test_res, test_errs], axis=1)

    test_res = pd.concat([test_res, test_res]).reset_index(drop=True)

    print(f'''
===========================================================
=================== Final Stats ===========================
===========================================================
Configuration:
    > {N_LAYERS} layers
    > {N_UNITS} units
    > {EPOCHS} epochs
    > Optimizer = {OPTIMIZER}
    > LR = {INIT_LR}
Mean Errors:   
{test_errs.mean()}
===========================================================
    ''')
