import datetime
import os
import pathlib

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

plt.style.use('ggplot')
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
        self.layers = []
        self._build()

    def _build(self):
        self.layers = [
            torch.nn.Linear(self.n_features, self.n_units),
            torch.nn.BatchNorm1d(self.n_units),
            torch.nn.SiLU()
        ]

        for lyr in range(self.n_layers):
            self.layers.append(torch.nn.Linear(self.n_units, self.n_units))
            self.layers.append(torch.nn.BatchNorm1d(self.n_units))
            self.layers.append(torch.nn.SiLU())

        self.layers.append(torch.nn.Linear(self.n_units, self.n_labels))
        self.layers.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


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


def run_train(model: torch.nn.Module, epochs: int, train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader, optimizer: torch.optim, loss_function: torch.nn, device: torch.device = torch.device('cpu')):
    train_losses = np.array([])
    val_losses = np.array([])
    for epch in range(epochs):
        print(f'Epoch: {epch}/{epochs} ({100 * epch / epochs:.2f}% done)')
        btch_train_losses = np.array([])
        btch_pbar = tqdm.tqdm(train_data_loader)
        for (X, Y) in btch_pbar:
            X = X.to(device)
            results = model(X)
            loss = loss_function(results, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            btch_train_losses = np.append(btch_train_losses, loss.item())
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

            # - Create the data frame
            btch_results = pd.DataFrame(d, index=pd.Index([0]))

            # - Add the batch data frame to the total results
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


def get_errors(results: pd.DataFrame, columns: list):
    n_columns = len(columns)
    true = results.iloc[:, :n_columns].values
    pred = results.iloc[:, n_columns:].values

    columns = [column_name + '_errors(%)' for column_name in columns]

    errors = pd.DataFrame(100 - true * 100 / pred, columns=columns)

    return errors


def run_ablation(data_root: pathlib.Path, features: list, labels: list, batch_size_numbers: list, epoch_numbers: list, layer_numbers: list, unit_numbers: list, root_save_dir: pathlib.Path):
    ablation_root_save_dir = root_save_dir / f'ablation_{TS}'
    os.makedirs(ablation_root_save_dir)

    ablation_res = pd.DataFrame()
    for root, data_folders, _ in os.walk(data_root):
        root = pathlib.Path(root)

        ablation_test_data_res = pd.DataFrame()
        for data_folder in tqdm.tqdm(data_folders):
            data_folder = root / data_folder
            for n_epochs in epoch_numbers:
                for n_batch_size in batch_size_numbers:
                    for n_layers in layer_numbers:
                        for n_units in unit_numbers:
                            # - Get the data
                            train_data_df = pd.read_csv(data_folder / 'train_data.csv')

                            # - Split into train / val
                            train_df, val_df = get_train_val_split(train_data_df, validation_proportion=0.2)

                            # - Dataset
                            train_ds, val_ds = QoEDataset(data=train_df, features=features, labels=labels), QoEDataset(data=val_df, features=features, labels=labels)

                            # - Data Loader
                            train_dl = torch.utils.data.DataLoader(
                                train_ds,
                                batch_size=n_batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=1
                            )

                            val_batch_size = n_batch_size // 4
                            val_dl = torch.utils.data.DataLoader(
                                val_ds,
                                batch_size=val_batch_size if val_batch_size > 0 else 1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=1
                            )

                            # - Build the model
                            mdl = QoEModel(n_features=len(features), n_labels=len(labels), n_layers=n_layers, n_units=n_units)
                            mdl.to(DEVICE)

                            # - Optimizer
                            optimizer = torch.optim.Adam(mdl.parameters())

                            # - Loss
                            loss_func = torch.nn.MSELoss()

                            # - Train
                            # - Create the train directory
                            train_save_dir = data_folder / f'{n_layers}_layers_{n_units}_units_{n_epochs}_epochs'
                            os.makedirs(train_save_dir)
                            train_losses, val_losses = run_train(
                                model=mdl,
                                epochs=n_epochs,
                                train_data_loader=train_dl,
                                val_data_loader=val_dl,
                                optimizer=optimizer,
                                loss_function=loss_func,
                                device=DEVICE
                            )

                            plt.plot(train_losses, label='train')
                            plt.plot(val_losses, label='val')
                            plt.suptitle('Train / Validation Loss Plot')
                            plt.legend()
                            plt.savefig(train_save_dir / 'train_val_loss.png')

                            test_data_df = pd.read_csv(data_folder / 'test_data.csv')
                            test_ds = QoEDataset(data=test_data_df, features=features, labels=labels)
                            test_dl = torch.utils.data.DataLoader(
                                test_ds,
                                batch_size=16,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=1
                            )

                            test_res = run_test(model=mdl, data_loader=test_dl, device=DEVICE)
                            test_res = unnormalize_results(results=test_res, data_set=test_ds, n_columns=len(test_res.columns) // 2)
                            test_res.to_csv(train_save_dir / f'test_results_{n_layers}_layers_{n_units}_units_{n_epochs}_epochs.csv', index=False)

                            test_errs = get_errors(results=test_res, columns=test_ds.labels.columns)
                            test_errs.to_csv(train_save_dir / f'test_errors_{n_layers}_layers_{n_units}_units_{n_epochs}_epochs.csv', index=False)

                            test_res = pd.concat([test_res, test_errs], axis=1)

                            ablation_test_data_res = pd.concat([ablation_test_data_res, test_res]).reset_index(drop=True)

                            print(f'''
                            ===========================================================
                            =================== Final Stats ===========================
                            ===========================================================
                            Configuration:
                                > {n_epochs} epochs
                                > {n_batch_size} batch size
                                > {n_layers} layers
                                > {n_units} units
                                > {n_epochs} epochs
                            Mean Errors:   
                            {test_errs.mean()}
                            ===========================================================
                                ''')
            ablation_test_data_res.to_csv(data_folder / 'results.csv', index=False)
            ablation_res = pd.concat([ablation_res, ablation_test_data_res]).reset_index(drop=True)
        ablation_res.to_csv(root_save_dir / 'ablation_results.csv', index=False)


N_LAYERS = 64
N_UNITS = 64
EPOCHS = 50
# EPOCHS = 100
TRAIN_BATCH_SIZE = 16
# TRAIN_BATCH_SIZE = 32


if __name__ == '__main__':
    # - Get the data
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
        num_workers=1
    )

    VAL_BATCH_SIZE = TRAIN_BATCH_SIZE // 4
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE if VAL_BATCH_SIZE > 0 else 1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    # - Build the model
    mdl = QoEModel(n_features=len(features), n_labels=len(labels), n_layers=N_LAYERS, n_units=N_UNITS)
    mdl.to(DEVICE)

    # - Optimizer
    optimizer = torch.optim.Adam(mdl.parameters())

    # - Loss
    loss_func = torch.nn.MSELoss()

    # - Train
    # - Create the train directory
    train_save_dir = SAVE_DIR / f'{N_LAYERS}_layers_{N_UNITS}_units_{EPOCHS}_epochs_{TS}'
    os.makedirs(train_save_dir)
    train_losses, val_losses = run_train(
        model=mdl,
        epochs=EPOCHS,
        train_data_loader=train_dl,
        val_data_loader=val_dl,
        optimizer=optimizer,
        loss_function=loss_func,
        device=DEVICE
    )

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.suptitle('Train / Validation Loss Plot')
    plt.legend()
    plt.savefig(train_save_dir / 'train_val_loss.png')

    test_data_df = pd.read_csv(DATA_ROOT / 'test_data.csv')
    test_ds = QoEDataset(data=test_data_df, features=features, labels=labels)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    test_res = run_test(model=mdl, data_loader=test_dl, device=DEVICE)
    test_res = unnormalize_results(results=test_res, data_set=test_ds, n_columns=len(test_res.columns)//2)
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
Mean Errors:   
{test_errs.mean()}
===========================================================
    ''')
