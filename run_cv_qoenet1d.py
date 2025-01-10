import os
import datetime
import pathlib

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm

from configs.params import VAL_PROP, LR_REDUCTION_FREQ, LR_REDUCTION_FCTR, DROPOUT_START, DROPOUT_P, OUTLIER_TH
from data_division import build_test_datasets
from models import QoENet1D
from regression_utils import eval_regressor, calc_errors
from utils.data_utils import get_train_val_split, QoEDataset
from utils.train_utils import run_train, run_test
import torch

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/Projects/PhD/QoE/Data/zoom/encrypted_traffic/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/Projects/PhD/QoE/Code/qoe')
OUTPUT_DIR = ROOT_OUTPUT_DIR / f'outputs_{TS}'


def min_max_norm(data: pd.DataFrame):
    data /= (data.max() - data.min())


# class QoEDataset(torch.utils.data.Dataset):
#     def __init__(self, data_df: pd.DataFrame, feature_columns: list, label_columns: list, remove_outliers: bool = False):
#         super().__init__()
#         self.data_df = data_df
#
#         self.rmv_outliers = remove_outliers
#
#         self.feature_columns = feature_columns
#         self.feature_df = None
#
#         self.label_columns = label_columns
#         self.label_df = None
#
#         self.labels_mu, self.labels_std = .0, .0
#
#         self.prepare_data()
#
#     def __len__(self):
#         return len(self.data_df)
#
#     def __getitem__(self, index):
#         return torch.as_tensor(self.feature_df.iloc[index], dtype=torch.float32), torch.as_tensor(self.label_df.iloc[index], dtype=torch.float32)
#
#     def prepare_data(self):
#         # 1) Drop unused columns
#         cols2drop = np.setdiff1d(list(self.data_df.columns), np.union1d(self.feature_columns, self.label_columns))
#         self.data_df = self.data_df.drop(columns=cols2drop)
#
#         # 2) Clean Na lines
#         self.data_df = self.data_df.loc[self.data_df.isna().sum(axis=1) == 0]
#
#         # 3) Outliers removal
#         if self.rmv_outliers:
#             self.data_df = self.remove_outliers(data_df=self.data_df, std_th=OUTLIER_TH)
#
#         # 4) Split to features and labels
#         self.feature_df = self.data_df.loc[:, self.feature_columns]
#
#         # 5) Standardize the features
#         self.feature_df, _, _ = self.standardize_data(self.feature_df)
#
#         # 6) Get the labels column
#         self.label_df = self.data_df.loc[:, self.label_columns]
#
#     @staticmethod
#     def standardize_data(data_df):
#         mu, std = data_df.mean(), data_df.std()
#         data_norm_df = (data_df - mu) / std
#         return data_norm_df, mu, std
#
#     @staticmethod
#     def remove_outliers(data_df: pd.DataFrame, std_th: int):
#
#         dataset_no_outliers = data_df.loc[(np.abs(scipy.stats.zscore(data_df)) < std_th).all(axis=1)]
#
#         L = len(data_df)
#         N = len(dataset_no_outliers)
#         R = 100 - N * 100 / L
#         print(f'''
#     Outliers
#         Total before reduction: {L}
#         Total after reduction: {N}
#         > Present reduced: {R:.3f}%
#     ''')
#
#         return dataset_no_outliers

def get_data(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, labels: list, batch_size: int = None, val_prop: float = None):
    train_data, val_data, test_data, test_ds = None, None, None, None
    if isinstance(batch_size, int) and isinstance(val_prop, float):
        # - Split into train / val
        train_df, val_df = get_train_val_split(
            train_df,
            validation_proportion=val_prop
        )

        # - Train dataloader
        train_data = torch.utils.data.DataLoader(
            QoEDataset(
                data_df=train_df,
                feature_columns=features,
                label_columns=labels,
                normalize_features=True,
                normalize_labels=True,
                remove_outliers=True
            ),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=1,
            drop_last=True
        )

        # - Validation dataloader
        val_batch_size = batch_size // 4
        val_data = torch.utils.data.DataLoader(
            QoEDataset(
                data_df=val_df,
                feature_columns=features,
                label_columns=labels,
                normalize_features=True,
                normalize_labels=True,
                remove_outliers=True
            ),
            batch_size=val_batch_size if val_batch_size > 0 else 1,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            drop_last=True
        )

        # - Test dataloader
        test_ds = QoEDataset(
            data_df=test_df,
            feature_columns=features,
            label_columns=labels,
            normalize_features=True,
            normalize_labels=True,
        )
        test_data = torch.utils.data.DataLoader(
            test_ds,
            batch_size=val_batch_size if val_batch_size > 0 else 1,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            drop_last=True
        )
    else:
        X_train = train_df.loc[:, features].values
        y_train = train_df.loc[:, labels].values
        train_data = (X_train, y_train)

        X_test = test_df.loc[:, features].values
        y_test = test_df.loc[:, labels].values
        test_data = (X_test, y_test)

    return train_data, val_data, test_data, test_ds

def train_model(model, epochs, train_data_loader, validation_data_loader, loss_function, optimizer, learning_rate, save_dir):
    # - Train
    # - Create the train directory
    train_save_dir = save_dir / f'train'
    os.makedirs(train_save_dir, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # - Train the model
    train_losses, val_losses = run_train(
        model=model,
        epochs=epochs,
        train_data_loader=train_data_loader,
        val_data_loader=validation_data_loader,
        loss_function=loss_function(),
        optimizer=optimizer(model.parameters(), lr=learning_rate),
        lr_reduce_frequency=LR_REDUCTION_FREQ,
        lr_reduce_factor=LR_REDUCTION_FCTR,
        dropout_epoch_start=DROPOUT_START,
        p_dropout_init=DROPOUT_P,
        device=device
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

def run_cv(model, data_df: pd.DataFrame, n_folds: int, features: list, labels: list, data_dir: pathlib.Path, output_dir: pathlib.Path or None, nn_params: dict):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cv_root_dir = data_dir / f'cv_{n_folds}_folds'
    if not cv_root_dir.is_dir():
        os.makedirs(cv_root_dir, exist_ok=True)
        build_test_datasets(
            data_df,
            n_folds=n_folds,
            root_save_dir=cv_root_dir
        )

    results = pd.DataFrame(columns=['true', 'predicted', 'error (%)'], dtype=np.float32)
    for cv_dir in os.listdir(cv_root_dir):
        if cv_dir[0] != '.':
            train_df = pd.read_csv(cv_root_dir / cv_dir / 'train_data.csv')
            test_df = pd.read_csv(cv_root_dir / cv_dir / 'test_data.csv')
            train_data, val_data, test_data, test_ds = get_data(
                train_df=train_df,
                test_df=test_df,
                features=features,
                labels=labels,
                batch_size=nn_params.get('batch_size'),
                val_prop=nn_params.get('val_prop')
            )

            if isinstance(val_data, torch.utils.data.DataLoader):
                # - For NN-based models

                # - Build the model
                mdl = model(
                    n_features=len(features),
                    n_labels=len(labels),
                    n_layers=nn_params.get('n_layers'),
                    n_units=nn_params.get('n_units')
                )

                train_model(
                    model=mdl,
                    epochs=nn_params.get('epochs'),
                    train_data_loader=train_data,
                    validation_data_loader=val_data,
                    loss_function=nn_params.get('loss_function'),
                    optimizer=nn_params.get('optimizer'),
                    learning_rate=nn_params.get('learning_rate'),
                    save_dir=output_dir
                )
                y_test, y_pred = predict(model=mdl, data_loader=test_data, device=device)
            else:
                # - For ML-based models
                X_train, y_train = train_data[0], train_data[1]
                model.fit(
                    X_train,
                    y_train
                )

                X_test, y_test = test_data[0], test_data[1]

                # - Predict the y_test labels based on X_test features
                y_test, y_pred = model.predict(X_test)

            errors = calc_errors(true=y_test, predicted=y_pred)
            results = pd.concat([
                results,
                pd.DataFrame(
                    {
                        'true': y_test.flatten(),
                        'predicted': y_pred.flatten(),
                        'error (%)': errors.flatten()
                    }
                )
            ], ignore_index=True)

    if isinstance(output_dir, pathlib.Path):
        results.to_csv(output_dir / 'final_results.csv')
    mean_error = results.loc[:, "error (%)"].mean()
    std_error = results.loc[:, "error (%)"].std()
    print(f'''
Mean Stats on {n_folds} CV for {labels}:
    Mean Errors (%)
    ---------------
    {mean_error:.2f}+/-{std_error:.3f}
    ''')
    return results


def predict(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device = torch.device('cpu')):
    y_true = np.array([])
    y_pred = np.array([])
    model.eval()
    with torch.no_grad():
        for (X, y) in tqdm(data_loader):
            # - Move the data to device
            X = X.to(device)
            y = y.to(device)

            # - Calculate predictions
            preds = model(X)

            # - Append to the output arrays
            y_true = np.append(y_true, y)
            y_pred = np.append(y_pred, preds)

    return y_true, y_pred
# NIQE, FPS, R
# FEATURES = ['BW', 'PPS']  # 2.50+/-2.294, 8.80+/-11.161, 9.03+/-13.215
# FEATURES = ['BW', 'ATP']  # 2.46+/-2.260, 8.75+/-10.937, 9.15+/-12.917
# FEATURES = ['BW', 'PL']  # 2.45+/-2.183, 8.83+/-11.085, 8.75+/-12.635
# FEATURES = ['BW', 'PPS', 'ATP']  # 2.48+/-2.282, 8.69+/-11.205, 8.97+/-13.059
# FEATURES = ['BW', 'ATP', 'PL']  # 2.44+/-2.186, 8.77+/-11.270, 8.49+/-12.276
# FEATURES = ['BW', 'PPS', 'PL']  # 2.39+/-2.183, 8.68+/-11.308, 8.67+/-12.547
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']  # 2.40+/-2.203, 8.79+/-11.147, 8.48+/-12.289

# FEATURES = ['L']  # 5.20+/-4.473, 16.07+/-35.003, 15.86+/-20.085
# FEATURES = ['J']  # 4.48+/-3.523, 13.12+/-13.391, 15.14+/-17.379
# FEATURES = ['L', 'J']  # 4.34+/-4.110, 13.87+/-20.041, 14.78+/-24.409

# FEATURES = ['PPS']  # 4.210+/-3.6881, 14.544+/-24.2634, 18.138+/-33.4549
# FEATURES = ['ATP']  # 4.060+/-3.6604, 13.511+/-22.4605, 17.179+/-29.0487
# FEATURES = ['PPS', 'ATP']

# FEATURES = ['BW', 'ATP', 'PL']  # NIQE best
FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']  # 3.14+/-2.815, 11.30+/-20.179, 10.89+/-23.477

LABELS = ['NIQE', 'FPS', 'R']
EPOCHS = 1
BATCH_SIZE = 64
N_LAYERS = 64
N_UNITS = 16
LOSS_FUNCTIONS = torch.nn.MSELoss
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 1e-3

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR)
    data_df = pd.read_csv(DATA_FILE)
    run_cv(
        model=QoENet1D,
        data_df=data_df,
        n_folds=10,
        features=FEATURES,
        labels=LABELS,
        data_dir=DATA_FILE.parent,
        output_dir=OUTPUT_DIR,
        nn_params=dict(
            batch_size=BATCH_SIZE,
            val_prop=VAL_PROP,
            n_layers=N_LAYERS,
            n_units=N_LAYERS,
            epochs=EPOCHS,
            loss_function=torch.nn.MSELoss,
            learning_rate=LEARNING_RATE,
            optimizer=torch.optim.Adam
        )
    )
