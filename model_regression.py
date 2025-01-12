import argparse
import datetime
import os
import pathlib

import numpy as np
import pandas as pd
import sklearn.decomposition
import torch
import tqdm
import scipy
import matplotlib
import matplotlib.pyplot as plt
EPSILON = 1e-9
matplotlib.use('Agg')
plt.style.use('ggplot')


class QoEDataset(torch.utils.data.Dataset):
    def __init__(self, data_df: pd.DataFrame, feature_columns: list, label_columns: list, pca: sklearn.decomposition.PCA = None, remove_outliers: bool = False):
        super().__init__()
        self.data_df = data_df

        self.rmv_outliers = remove_outliers

        self.feature_columns = feature_columns
        self.feature_df = None

        self.label_columns = label_columns
        self.label_df = None

        self.labels_mu, self.labels_std = .0, .0

        self.pca = pca

        self.prepare_data()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return torch.as_tensor(self.feature_df.iloc[index], dtype=torch.float32), torch.as_tensor(self.label_df.iloc[index], dtype=torch.float32)

    def prepare_data(self):
        # 1) Drop unused columns
        cols2drop = np.setdiff1d(list(self.data_df.columns), np.union1d(self.feature_columns, self.label_columns))
        self.data_df = self.data_df.drop(columns=cols2drop)

        # 2) Clean Na lines
        self.data_df = self.data_df.loc[self.data_df.isna().sum(axis=1) == 0]

        # 3) Outliers removal
        if self.rmv_outliers:
            self.data_df = self.remove_outliers(data_df=self.data_df, std_th=OUTLIER_TH)

        # 4) Split to features and labels
        self.feature_df = self.data_df.loc[:, self.feature_columns]
        self.label_df = self.data_df.loc[:, self.label_columns]

        # 4) Normalization on features and labels separately, to be able in test time to distinguish between them easily
        self.feature_df, _, _ = self.normalize(self.feature_df)
        self.label_df, self.labels_mu, self.labels_std = self.normalize(self.label_df)

        # 5) PCA on the features
        if isinstance(self.pca, sklearn.decomposition.PCA):
            self.feature_df = pd.DataFrame(np.dot(self.feature_df - self.pca.mean_, self.pca.components_.T))

    @staticmethod
    def normalize(data_df):
        mu, std = data_df.mean(), data_df.std()
        data_norm_df = (data_df - mu) / std
        return data_norm_df, mu, std

    @staticmethod
    def remove_outliers(data_df: pd.DataFrame, std_th: int):

        dataset_no_outliers = data_df.loc[(np.abs(scipy.stats.zscore(data_df)) < std_th).all(axis=1)]

        L = len(data_df)
        N = len(dataset_no_outliers)
        R = 100 - N * 100 / L
        print(f'''
    Outliers
        Total before reduction: {L}
        Total after reduction: {N}
        > Present reduced: {R:.3f}%
    ''')

        return dataset_no_outliers

    def unnormalize_labels(self, x):
        return x * self.labels_std + self.labels_mu


class QoEModel2(torch.nn.Module):
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
        in_units = self.n_features
        out_units = self.n_units
        # - Down path
        while out_units >= self.n_features:
            # - Add a layer
            self._add_layer(n_in=in_units, n_out=out_units, activation=torch.nn.SiLU)
            in_units = out_units
            out_units //= 2
            if out_units < self.n_features:
                out_units = self.n_features
                break

        while out_units < self.n_units:
            in_units = out_units
            out_units *= 2
            if out_units > self.n_units:
                out_units = self.n_units
                break
            # - Add a layer
            self._add_layer(n_in=in_units, n_out=out_units, activation=torch.nn.SiLU)

        self._add_layer(n_in=out_units, n_out=self.n_labels, activation=torch.nn.Tanh)

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x, p_drop: float = 0.0):
        x = self.model(x)

        x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)

        return x


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


def get_train_val_split(data: pd.DataFrame, validation_proportion: float = 0.2):
    n_data = len(data)
    data_indices = np.arange(n_data)

    n_val_items = int(n_data * validation_proportion)
    val_indices = np.random.choice(data_indices, n_val_items, replace=True)
    val_data = data.iloc[val_indices]

    train_indices = np.setdiff1d(data_indices, val_indices)
    train_data = data.iloc[train_indices]

    return train_data, val_data


def run_train(model: torch.nn.Module, epochs: int, train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader,
              loss_function: torch.nn, optimizer: torch.optim, lr_reduce_frequency: int, lr_reduce_factor: float = 1.0, dropout_epoch_start: int = 20, p_dropout_init: float = 0.1, device: torch.device = torch.device('cpu')):
    train_losses = np.array([])
    val_losses = np.array([])
    for epch in range(epochs):
        model.train(True)
        p_drop = p_dropout_init * (epch // dropout_epoch_start)
        print(f'Epoch: {epch}/{epochs} ({100 * epch / epochs:.2f}% done)')
        print(f'\t ** INFO ** p_drop = {p_drop:.4f}')
        btch_train_losses = np.array([])
        btch_pbar = tqdm.tqdm(train_data_loader)
        for (X, Y) in btch_pbar:
            X = X.to(device)
            Y = Y.to(device)
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
        model.eval()
        with torch.no_grad():
            for (X, Y) in val_data_loader:
                X = X.to(device)
                Y = Y.to(device)
                results = model(X)
                loss = loss_function(results, Y)
                btch_val_losses = np.append(btch_val_losses, loss.item())

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
        Y = Y.to(device)
        btch_preds = model(X)

        for y, pred in zip(Y, btch_preds):
            d = dict()

            # - Add labels
            for i, y_val in enumerate(y):
                d[f'true_{i}'] = np.float32(y_val.cpu().numpy())

            # - Add preds
            for i, pred_val in enumerate(pred):
                d[f'pred_{i}'] = np.float32(pred_val.detach().cpu().numpy())

            # - Create the cv_5_folds frame
            btch_results = pd.DataFrame(d, index=pd.Index([0]))

            # - Add the batch cv_5_folds frame to the total results
            test_results = pd.concat([test_results, btch_results])

    test_results = test_results.reset_index(drop=True)

    return test_results


def run_pca(dataset_df: pd.DataFrame):
    pca = sklearn.decomposition.PCA(n_components=dataset_df.shape[1])
    dataset_pca_df = pca.fit_transform(dataset_df)

    return dataset_pca_df, pca


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
        for old_name in data_set.data_df.columns:
            new_name = f'{old_name}_true'
            labels = labels.rename(columns={old_name: new_name})

        # - Get the predictions
        preds = pd.DataFrame(data_set.unnormalize_labels(res.iloc[0, n_columns:].values)).T
        for old_name in data_set.data_df.columns:
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

    errors = pd.DataFrame(np.abs(100 - true * 100 / (pred + EPSILON)), columns=columns)

    return errors


def run_ablation(test_data_root: pathlib.Path, data_dirs: list, features: list, labels: list, batch_size_numbers: list, epoch_numbers: list, layer_numbers: list, unit_numbers: list, loss_functions: list, optimizers: list,
                 initial_learning_rates: list, save_dir: pathlib.Path):
    # - Will hold the final results
    ablation_results = pd.DataFrame()

    # - Will hold the metadata for all experiments
    test_metadata = pd.DataFrame()
    test_dir_names = os.listdir(test_data_root)
    test_dir_names = [test_dir_name for test_dir_name in test_dir_names if test_dir_name in data_dirs]

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
        train_data_df = train_data_df.loc[~train_data_df.isna().loc[:, 'Bandwidth']]
        _, train_pca = run_pca(dataset_df=train_data_df.loc[:, features])
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

                                    train_df, val_df = get_train_val_split(train_data_df, validation_proportion=VAL_PROP)

                                    # - Dataset
                                    train_ds = QoEDataset(
                                        data_df=train_df,
                                        feature_columns=features,
                                        label_columns=labels,
                                        remove_outliers=True,
                                        pca=train_pca
                                    )
                                    val_ds = QoEDataset(
                                        data_df=val_df,
                                        feature_columns=features,
                                        label_columns=labels,
                                        remove_outliers=True,
                                        pca=train_pca
                                    )

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
                                    # mdl = QoEModel2(n_features=len(features), n_labels=len(labels), n_layers=n_layers, n_units=n_units)
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
                                        lr_reduce_frequency=LR_REDUCTION_FREQ,
                                        lr_reduce_factor=LR_REDUCTION_FCTR,
                                        dropout_epoch_start=DROPOUT_START,
                                        p_dropout_init=DROPOUT_P,
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
                                    test_ds = QoEDataset(
                                        data_df=test_data_df,
                                        feature_columns=features,
                                        label_columns=labels,
                                        pca=train_pca,
                                    )
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
                                    test_errs = get_errors(results=test_res, columns=test_ds.label_columns)

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
                                    configuration_results = pd.concat([configuration_results, pd.DataFrame(test_errs.abs().mean()).T], axis=1)

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

Mean Errors (%):
{test_errs.abs().mean().values[0]}+/-{test_errs.abs().std().values[0]}

Status:
    - Experiment {exp_idx}/{total_exp} ({100 * exp_idx / total_exp:.2f}% done)
===========================================================
                                    ''')

    # - Save the final metadata and the results
    ablation_results.to_csv(save_dir / 'ablation_final_results.csv', index=False)
    print('***************************************************************')
    print(f'           {cv_folds}-Folds CV Final Results')
    print('***************************************************************')
    for lbl in labels:
        err_vals = ablation_results.loc[:, f'{lbl}_errors(%)'].values
        print(f'\t> {lbl} Mean {cv_folds}-fold CV Errors (%) based on {features}: {err_vals.mean():.2f}+/-{err_vals.std():.3f}')
    print('***************************************************************')
    # - Save the final metadata and the results
    test_metadata.to_csv(save_dir / 'test_metadata.csv', index=False)


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--n_layers', type=int, default=N_LAYERS, help='Number of NN-Blocks in the network')
    parser.add_argument('--n_units', type=int, default=N_UNITS, help='Number of units in each NN-Block')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--outlier_th', type=int, default=OUTLIER_TH, help='Represents the number of STDs from the mean to remove samples with')
    parser.add_argument('--lr', type=int, default=LR, help='Represents the initial learning rate of the optimizer')
    parser.add_argument('--lr_reduction_freq', type=int, default=LR_REDUCTION_FREQ, help='Represents the number of epochs for the LR reduction')
    parser.add_argument('--lr_reduction_fctr', type=float, default=LR_REDUCTION_FCTR, help='Represents the factor by which the LR reduced each LR_REDUCTION_FREQ epochs')
    parser.add_argument('--dropout_start', type=int, default=DROPOUT_START, help='The epoch when the dropout technique start being applied')
    parser.add_argument('--dropout_p', type=float, default=DROPOUT_P, help='The probability of the unit to be zeroed out')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--val_prop', type=float, default=VAL_PROP, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--train_data_file', type=str, default=TRAIN_DATA_FILE, help='The path to the train data file')
    parser.add_argument('--test_data_file', type=str, default=TEST_DATA_FILE, help='The path to the test data file')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')

    return parser


TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
N_LAYERS = 32
N_UNITS = 256
EPOCHS = 100
OPTIMIZER = torch.optim.Adam
LR = 0.01
LR_REDUCTION_FREQ = 20
LR_REDUCTION_FCTR = 0.5
DROPOUT_START = 50
DROPOUT_P = 0.1
BATCH_SIZE = 128
OUTLIER_TH = 2
VAL_PROP = 0.2

# FEATURES = ['Bandwidth', 'pps', 'Jitter', 'packets length', 'Interval start', 'Latency', 'avg time between packets']
FEATURES = ['Bandwidth', 'pps', 'packets length', 'avg time between packets']
# LABELS = ['NIQE']
LABELS = ['NIQE', 'Resolution', 'fps']

LOSS_FUNCTION = torch.nn.MSELoss()
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
TRAIN_DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/test/train_data.csv')
# TRAIN_DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/data_norm_no_outliers_pca.csv')
TEST_DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/test/test_data.csv')
OUTPUT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/test/output')


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # - Get the cv_5_folds
    train_data_df = pd.read_csv(TRAIN_DATA_FILE)

    # train_data_df.loc[:, FEATURES], train_pca = run_pca(dataset_df=train_data_df.loc[:, FEATURES])
    train_pca = None

    train_df, val_df = get_train_val_split(train_data_df, validation_proportion=args.val_prop)

    # - Dataset
    train_ds = QoEDataset(
        data_df=train_df,
        feature_columns=FEATURES,
        label_columns=LABELS,
        pca=train_pca,
        # remove_outliers=False,
        remove_outliers = True,
    )
    val_ds = QoEDataset(
        data_df=val_df,
        feature_columns=FEATURES,
        label_columns=LABELS,
        pca=train_pca,
        # remove_outliers=False,
        remove_outliers=True,
    )

    # - Data Loader
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        drop_last=True
    )

    VAL_BATCH_SIZE = args.batch_size // 4
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE if VAL_BATCH_SIZE > 0 else 1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=True
    )

    # - Build the model
    mdl = QoEModel2(n_features=len(FEATURES), n_labels=len(LABELS), n_layers=args.n_layers, n_units=args.n_units)
    mdl.to(DEVICE)

    # - Optimizer
    optimizer = OPTIMIZER(mdl.parameters(), lr=args.lr)

    # - Loss
    loss_func = LOSS_FUNCTION

    # - Train
    # - Create the train directory
    train_save_dir = args.output_dir / f'{args.n_layers}_layers_{args.n_units}_units_{args.epochs}_epochs_{TS}'
    os.makedirs(train_save_dir)
    train_losses, val_losses = run_train(
        model=mdl,
        epochs=args.epochs,
        train_data_loader=train_dl,
        val_data_loader=val_dl,
        loss_function=loss_func,
        optimizer=optimizer,
        lr_reduce_frequency=args.lr_reduction_freq,
        lr_reduce_factor=args.lr_reduction_fctr,
        dropout_epoch_start=args.dropout_start,
        p_dropout_init=args.dropout_p,
        device=DEVICE
    )

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.suptitle('Train / Validation Loss Plot')
    plt.legend()
    plt.savefig(train_save_dir / 'train_val_loss.png')
    plt.close()

    test_data_df = pd.read_csv(TEST_DATA_FILE)
    test_ds = QoEDataset(
        data_df=test_data_df,
        feature_columns=FEATURES,
        label_columns=LABELS,
        pca=train_pca,
        remove_outliers=False
    )
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
    test_res.to_csv(train_save_dir / f'test_results_{args.n_layers}_layers_{args.n_units}_units_{args.epochs}_epochs.csv')

    test_errs = get_errors(results=test_res, columns=test_ds.label_columns)
    test_errs.to_csv(train_save_dir / f'test_errors_{args.n_layers}_layers_{args.n_units}_units_{args.epochs}_epochs.csv')

    test_res = pd.concat([test_res, test_errs], axis=1)

    test_res = pd.concat([test_res, test_res]).reset_index(drop=True)

    print(f'''
===========================================================
=================== Final Stats ===========================
===========================================================
Configuration:
    > {args.n_layers} layers
    > {args.n_units} units
    > {args.epochs} epochs
    > Optimizer = {OPTIMIZER}
    > LR = {args.lr}
Mean Errors:
{test_errs.mean()}
===========================================================
    ''')
