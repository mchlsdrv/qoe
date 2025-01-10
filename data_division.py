import os
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm


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


DATA_NAME = 'niqe_rbm'
N_FOLDS = 10
DATA_ROOT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/rbm')
DATA_SET_PATH = DATA_ROOT_DIR / f'niqe_data_clean_rbm_float.csv'
SAVE_DIR = DATA_ROOT_DIR / f'{DATA_NAME}_cv_{N_FOLDS}_folds_float'

if __name__ == '__main__':
    data_set = pd.read_csv(DATA_SET_PATH)
    if 'Unnamed: 0' in data_set.columns:
        DATA_SET = data_set.drop(columns=['Unnamed: 0'])
    build_test_datasets(data=data_set, n_folds=N_FOLDS, root_save_dir=SAVE_DIR)
