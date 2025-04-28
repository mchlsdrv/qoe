import os
import pathlib

import torch
import torch.utils.data
import numpy as np
import pandas as pd
import sklearn
import sklearn.decomposition
import scipy
from tqdm import tqdm
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
        X, Y = self.feature_df.iloc[index].values, self.label_df.iloc[index].values
        if self.tokenize:
            toks = self.tocknzr(str(X), padding='max_length', truncation=True)
            X, att_msk = toks.get('input_ids'), toks.get('attention_mask')
            return torch.as_tensor(X, dtype=torch.int64), torch.as_tensor(att_msk, dtype=torch.int64), torch.as_tensor(Y, dtype=torch.float32)
        return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)

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


def build_test_datasets(data: pd.DataFrame, n_folds: int,  root_save_dir: pathlib.Path):
    """
    Divides the cv_n_folds into n_test_sets train-train_test datasets each with proportion of
    (1-test_set_proportion):test_set_proportion respectively.

    * Each train_test dataset is chosen to not include items from other train_test sets

    :param data: pandas.DataFrame object containing the cv_5_folds
    :param n_folds: Number of train_test sets to produce
    :param root_save_dir: The location to save the datasets at
    :return: None
    """
    # - Get the total number of items
    n_items = len(data)

    # - Get the number of train_test items
    test_set_proportion = n_folds / 100
    n_test_items = int(n_items * test_set_proportion)

    # - Produce the total set of indices
    all_idxs = np.arange(n_items)

    # - Produce the set of indices which may be chosen for train_test
    valid_test_idxs = np.arange(n_items)

    # - Create an n_test_sets train-train_test sets
    for test_set_idx in tqdm(range(n_folds)):
        # - Chose randomly n_test_items from the all_idxs
        test_idxs = np.random.choice(all_idxs, n_test_items, replace=False)

        # - Update the valid_test_idxs by removing the once which were chosen for the train_test
        valid_test_idxs = np.setdiff1d(valid_test_idxs, test_idxs)

        # - Get the train_test items
        test_data = data.iloc[test_idxs].reset_index(drop=True)

        # - Get the train_test items
        train_data = data.iloc[np.setdiff1d(all_idxs, test_idxs)].reset_index(drop=True)

        # - Save the train / train_test datasets

        # -- Create a dir for the current train_test set
        test_set_save_dir = root_save_dir / f'train_test{test_set_idx}'
        os.makedirs(test_set_save_dir, exist_ok=True)

        # -- Save teh datasets
        # train_data.dropna(axis=0, inplace=True)
        train_data.to_csv(test_set_save_dir / f'train_data.csv', index=False)

        # test_data.dropna(axis=0, inplace=True)
        test_data.to_csv(test_set_save_dir / f'test_data.csv', index=False)


def remove_outliers(dataset: pd.DataFrame, columns: list, std_th: int, log_file):
    L = len(dataset)
    dataset = dataset.loc[(np.abs(scipy.stats.zscore(dataset.loc[:, columns])) < std_th).all(axis=1)]
    N = len(dataset)
    reduct_pct = calc_data_reduction(L, N)
    print(f'''
Outliers
    Total before reduction: {L}
    Total after reduction: {N}
    > Present reduced: {reduct_pct:.3f}%
''', file=log_file)

    return dataset, reduct_pct


def remove_zero_labels(dataset: pd.DataFrame, labels: list):
    # - Clean the data points where labels are equal to 0, as it is not a realistic
    for lbl in labels:
        dataset = dataset.loc[dataset.loc[:, lbl] > 0]


def normalize_columns(data_df, columns):
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


def get_data(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, labels: list, batch_size: int = None, val_prop: float = None, tokenize: bool = False):
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
                tokenize=tokenize
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
                tokenize=tokenize
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
            tokenize=tokenize
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


def get_input_data(data, tokenize: bool, device: torch.cuda.device):
    if tokenize:
        X, att_msk, Y = data
        X = X.to(device)
        Y = Y.to(device)
        att_msk = att_msk.to(device)

        input_data = [X, att_msk]
    else:
        X, Y = data
        X = X.to(device)
        Y = Y.to(device)

        input_data = [X]

    return input_data, Y

