import os
import datetime
import pathlib

import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib import pyplot as plt
from tqdm import tqdm

from configs.params import VAL_PROP, LR_REDUCTION_FREQ, LR_REDUCTION_FCTR, DROPOUT_START, DROPOUT_P, EPSILON, OUTLIER_TH
from models import QoENet1D
from regression_utils import calc_errors, normalize_columns
from utils.data_utils import get_train_val_split
from utils.train_utils import run_train
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig


class QoEDataset(torch.utils.data.Dataset):
    def __init__(self, data_df: pd.DataFrame, feature_columns: list, label_columns: list, normalize_features: bool = False, normalize_labels: bool = False, pca=None, remove_outliers: bool = False):
        super().__init__()
        self.tocknzr = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.data_df = data_df

        self.rmv_outliers = remove_outliers

        self.feature_columns = feature_columns
        self.feature_df = None

        self.label_columns = label_columns
        self.label_df = None

        self.labels_mu, self.labels_std = .0, .0

        self.pca = pca

        self.normalize_features = normalize_features

        self.normalize_labels = normalize_labels

        self.prepare_data()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        X, y = self.feature_df.iloc[index].values, self.label_df.iloc[index].values
        tocks = self.tocknzr([str(X)], padding='max_length', truncation=True)
        X, att_msk = tocks.get('input_ids'), tocks.get('attention_mask')
        return torch.as_tensor(X, dtype=torch.int64), torch.as_tensor(att_msk, dtype=torch.int64), torch.as_tensor(y, dtype=torch.float32)

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
        if self.normalize_features:
            self.feature_df, _, _ = self.normalize_data(self.feature_df)

        self.label_df = self.data_df.loc[:, self.label_columns]
        if self.normalize_labels:
            self.label_df, self.labels_mu, self.labels_std = self.normalize_data(self.label_df)

        # 5) PCA on the features
        if isinstance(self.pca, sklearn.decomposition.PCA):
            self.feature_df = pd.DataFrame(np.dot(self.feature_df - self.pca.mean_, self.pca.components_.T))

    @staticmethod
    def normalize_data(data_df):
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


class TransformerForRegression(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.regressor = torch.nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.regressor(pooled_output)


def min_max_norm(data: pd.DataFrame):
    data /= (data.max() - data.min() + EPSILON)


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
                normalize_labels=False,
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
                normalize_labels=False,
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
            normalize_labels=False,
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


def calc_reduction(original_size, reduced_size):
    reduction_pct = 100 - 100 * reduced_size / original_size
    return reduction_pct


def run_cv(model, cv_root_dir: pathlib.Path, n_folds: int, features: list, labels: list, output_dir: pathlib.Path or None, nn_params: dict):
    train_data_reductions = np.array([])
    test_data_reductions = np.array([])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results = pd.DataFrame(columns=['true', 'predicted', 'error (%)'], dtype=np.float32)
    for fold_dir in os.listdir(cv_root_dir):
        if fold_dir[0] != '.':

            feat_lbls_names = [*features, *labels]

            # - Train data
            train_df = pd.read_csv(cv_root_dir / fold_dir / 'train_data.csv')
            train_df = train_df.loc[:, feat_lbls_names]
            train_data_len_orig = len(train_df)

            # -- Remove the test data rows where the label is 0
            train_df = train_df.loc[train_df.loc[:, *labels] > 0]
            train_data_len_reduced = len(train_df)
            train_rdct_pct = calc_reduction(original_size=train_data_len_orig, reduced_size=train_data_len_reduced)
            train_data_reductions = np.append(train_data_reductions, train_rdct_pct)

            train_df, _, _ = normalize_columns(data_df=train_df, columns=[*features])

            # - Test data
            test_df = pd.read_csv(cv_root_dir / fold_dir / 'test_data.csv')
            test_df = test_df.loc[:, feat_lbls_names]
            test_data_len_orig = len(test_df)

            # -- Remove the test data rows where the label is 0
            test_df = test_df.loc[test_df.loc[:, *labels] > 0]
            test_data_len_reduced = len(test_df)
            test_rdct_pct = calc_reduction(original_size=test_data_len_orig, reduced_size=test_data_len_reduced)
            test_data_reductions = np.append(test_data_reductions, test_rdct_pct)

            test_df, _, _ = normalize_columns(data_df=test_df, columns=[*features])

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
                mdl = model(model_name='bert-base-uncased')

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
    
    Mean reduced data (%)
        - Train: {train_data_reductions.mean():.2f}+/-{train_data_reductions.std():.3f}
        - Test: {test_data_reductions.mean():.2f}+/-{test_data_reductions.std():.3f}
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
            y_true = np.append(y_true, y.cpu().numpy())
            y_pred = np.append(y_pred, preds.cpu().numpy())

    return y_true, y_pred


DATA_TYPE = 'packet_size'
# DATA_TYPE = 'piat'

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

CV_ROOT_DIR = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\qoe\\whatsapp\\data\\packet_size_cv_10_folds_float')
OUTPUT_DIR = pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\qoe\\whatsapp\\output\\cv_{TS}')
os.makedirs(OUTPUT_DIR)

PAKET_SIZE_FEATURES = [
    'number_of_packet_sizes_in_time_window',
    'number_of_unique_packet_sizes_in_time_window',
    'min_packet_size',
    'max_packet_size',
    'mean_packet_size',
    'std_packet_size',
    'q1_packet_size',
    'q2_packet_size',
    'q3_packet_size',
]

PIAT_FEATURES = [
    'number_of_piats_in_time_window',
    'number_of_unique_piats_in_time_window',
    'min_piat',
    'max_piat',
    'mean_piat',
    'std_piat',
    'q1_piat',
    'q2_piat',
    'q3_piat',
]
# LABELS = ['brisque']
# LABELS = ['piqe']
LABELS = ['fps']
# LABELS = ['brisque', 'piqe', 'fps']
EPOCHS = 500
BATCH_SIZE = 32
N_LAYERS = 32
N_UNITS = 512
LOSS_FUNCTIONS = torch.nn.MSELoss
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 1e-3
MODEL = TransformerForRegression

if __name__ == '__main__':
    run_cv(
        model=MODEL,
        n_folds=10,
        features=PAKET_SIZE_FEATURES if DATA_TYPE == 'packet_size' else PIAT_FEATURES,
        labels=LABELS,
        cv_root_dir=CV_ROOT_DIR,
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
