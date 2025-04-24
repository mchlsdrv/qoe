import os
import pathlib
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

from configs.params import PACKET_SIZE_FEATURES, PIAT_FEATURES
from utils.data_utils import calc_data_reduction


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
        train_data.to_csv(test_set_save_dir / f'train_data.csv', index=False)
        test_data.to_csv(test_set_save_dir / f'test_data.csv', index=False)


def remove_outliers(dataset: pd.DataFrame, columns: list, std_th: int):
    L = len(dataset)
    dataset = dataset.loc[(np.abs(scipy.stats.zscore(dataset.loc[:, columns])) < std_th).all(axis=1)]
    N = len(dataset)
    data_reduct = calc_data_reduction(L, N)
    print(f'''
Outliers
    Total before reduction: {L}
    Total after reduction: {N}
    > Present reduced: {data_reduct:.3f}%
''')

    return dataset


def remove_zero_labels(dataset: pd.DataFrame, labels: list):
    # - Clean the data points where labels are equal to 0, as it is not a realistic
    for lbl in labels:
        dataset = dataset.loc[dataset.loc[:, lbl] > 0]


DATA_NAME = 'piat'
# DATA_NAME = 'packet_size'
LABELS = ['brisque', 'piqe', 'fps']
N_FOLDS = 10
DATA_ROOT_DIR = pathlib.Path(f'/home/projects/bagon/msidorov/projects/qoe/whatsapp/data')
DATA_SET_PATH = DATA_ROOT_DIR / f'{DATA_NAME}_features_labels.csv'
SAVE_DIR = DATA_ROOT_DIR / f'{DATA_NAME}_cv_{N_FOLDS}_folds_float'
OUTLIER_STD_TH = 3

if __name__ == '__main__':
    data_df = pd.read_csv(DATA_SET_PATH)
    if 'Unnamed: 0' in data_df.columns:
        DATA_SET = data_df.drop(columns=['Unnamed: 0'])

    data_df = remove_outliers(
        dataset=data_df,
        columns=PIAT_FEATURES if DATA_NAME == 'piat' else PACKET_SIZE_FEATURES,
        std_th=OUTLIER_STD_TH
    )

    # - Leave only the rows where the labels are > 0
    # data_df = data_df.loc[(data_df.loc[:, LABELS] > 0).prod(axis=1).astype(bool)].reset_index(drop=True)

    build_test_datasets(data=data_df, n_folds=N_FOLDS, root_save_dir=SAVE_DIR)
