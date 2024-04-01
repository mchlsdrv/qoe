import pathlib
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader


DATA_PATH = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/data/Zoom/Encrypted-Zoom-traffic-dataset-main/Dataset.csv')


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
        return x.numpy() * self.features_std + self.features_mu

    def unnormalize_labels(self, y):
        return y.numpy() * self.labels_std + self.labels_mu


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


def train(model: torch.nn.Module, epochs: int, train_data_loader: torch.utils.data.DataLoader, val_data_loader: torch.utils.data.DataLoader, optimizer: torch.optim, loss_function: torch.nn, device: torch.device = torch.device('cpu')):
    train_losses = np.array([])
    for epch in range(epochs):
        print(f'Epoch: {epch}/{epochs} ({100 * epch / epochs:.2f}% done)')

        btch_pbar = tqdm.tqdm(train_data_loader)
        for (X, Y) in btch_pbar:
            X = X.to(device)
            results = model(X)
            loss = loss_function(results, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses = np.append(train_losses, loss.item())

        val_losses = np.array([])
        with torch.no_grad():
            model.eval()
            for (X, Y) in val_data_loader:
                X = X.to(device)
                results = model(X)
                loss = loss_function(results, Y)
                val_losses = np.append(val_losses, loss.item())

            model.train()

        print(f'''
        ===
        > Stats: train = {train_losses.mean():.4f}, val = {val_losses.mean():.4f}
        ===
        ''')

    return model


if __name__ == '__main__':
    # - Get the data
    data_df = pd.read_csv(DATA_PATH)

    train_df, val_df = get_train_val_split(data_df, validation_proportion=0.2)

    features = ['Bandwidth', 'pps', 'avg time between packets', 'packets length']
    labels = ['NIQE', 'Resolution', 'fps', 'Latancy', 'Jitter']

    # - Dataset
    train_ds, val_ds = QoEDataset(data=train_df, features=features, labels=labels), QoEDataset(data=val_df, features=features, labels=labels)

    # - Data Loader
    TRAIN_BATCH_SIZE = 16
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
    mdl = QoEModel(n_features=4, n_labels=5, n_layers=8, n_units=32)

    # - Optimizer
    optimizer = torch.optim.Adam(mdl.parameters())

    # - Loss
    loss_func = torch.nn.MSELoss()

    # - Train
    EPOCHS = 10
    mdl_trained = train(
        model=mdl,
        epochs=10,
        train_data_loader=train_dl,
        val_data_loader=val_dl,
        optimizer=optimizer,
        loss_function=loss_func,
        device=DEVICE
    )

