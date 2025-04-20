import torch
import torch.utils.data
import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import scipy
from transformers import AutoTokenizer

from configs.params import OUTLIER_TH, EPSILON


class QoEDataset(torch.utils.data.Dataset):
    def __init__(self, data_df: pd.DataFrame, feature_columns: list, label_columns: list, normalize_features: bool = False, normalize_labels: bool = False, pca=None, remove_outliers: bool = False, tokenize: bool = False):
        super().__init__()
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

        self.tokenize = tokenize
        if self.tokenize:
            self.tocknzr = AutoTokenizer.from_pretrained('bert-base-uncased')

        self.prepare_data()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        X, y = self.feature_df.iloc[index].values, self.label_df.iloc[index].values
        if self.tokenize:
            tocks = self.tocknzr(str(X), padding='max_length', truncation=True)
            X, att_msk = tocks.get('input_ids'), tocks.get('attention_mask')
            return torch.as_tensor(X, dtype=torch.int64), torch.as_tensor(att_msk, dtype=torch.int64), torch.as_tensor(y, dtype=torch.float32)
        return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)

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
        data_reduct = calc_data_reduction(L, N)
        print(f'''
    Outliers
        Total before reduction: {L}
        Total after reduction: {N}
        > Present reduced: {data_reduct:.3f}%
    ''')

        return dataset_no_outliers

    def unnormalize_labels(self, x):
        return x * self.labels_std + self.labels_mu


def normalize_columns(data_df, columns):
    data_df = data_df.astype(float)
    mu, std = data_df.loc[:, columns].mean(), data_df.loc[:, columns].std()
    data_df.loc[:, columns] = (data_df.loc[:, columns] - mu) / (std + EPSILON)
    return data_df, mu, std


def get_train_val_split(data: pd.DataFrame, validation_proportion: float = 0.2):
    n_data = len(data)
    data_indices = np.arange(n_data)

    n_val_items = int(n_data * validation_proportion)
    val_indices = np.random.choice(data_indices, n_val_items, replace=True)
    val_data = data.iloc[val_indices]

    train_indices = np.setdiff1d(data_indices, val_indices)
    train_data = data.iloc[train_indices]

    return train_data, val_data


def calc_data_reduction(original_size, reduced_size):
    reduction_pct = 100 - 100 * reduced_size / original_size
    return reduction_pct


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
                remove_outliers=True,
                tokenize=True
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
                remove_outliers=True,
                tokenize=True
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
            tokenize=True
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
