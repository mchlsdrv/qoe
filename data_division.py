import os
import pathlib
import numpy as np
import pandas as pd


def build_test_datasets(data: pd.DataFrame, n_test_sets, test_set_proportion: float, root_save_dir: pathlib.Path):
    """
    Divides the cv_5_folds into n_test_sets train-test datasets each with proportion of
    (1-test_set_proportion):test_set_proportion respectively.

    * Each test dataset is chosen to not include items from other test sets

    :param data: pandas.DataFrame object containing the cv_5_folds
    :param n_test_sets: Number of test sets to produce
    :param test_set_proportion: The proportion by which to divide the cv_5_folds
    :param root_save_dir: The location to save the datasets at
    :return: None
    """
    # - Get the total number of items
    n_items = len(data)

    # - Get the number of test items
    n_test_items = int(n_items * test_set_proportion)

    # - Produce the total set of indices
    all_idxs = np.arange(n_items)

    # - Produce the set of indices which may be chosen for test
    valid_test_idxs = np.arange(n_items)

    # - Create an n_test_sets train-test sets
    for test_set_idx in range(n_test_sets):
        # - Chose randomly n_test_items from the all_idxs
        test_idxs = np.random.choice(all_idxs, n_test_items, replace=False)

        # - Update the valid_test_idxs by removing the once which were chosen for the test
        valid_test_idxs = np.setdiff1d(valid_test_idxs, test_idxs)

        # - Get the test items
        test_data = data.iloc[test_idxs].reset_index(drop=True)

        # - Get the test items
        train_data = data.iloc[np.setdiff1d(all_idxs, test_idxs)].reset_index(drop=True)

        # - Save the train / test datasets

        # -- Create a dir for the current test set
        test_set_save_dir = root_save_dir / f'ablation/test{test_set_idx}'
        os.makedirs(test_set_save_dir, exist_ok=True)

        # -- Save teh datasets
        train_data.to_csv(test_set_save_dir / f'train_data.csv', index=False)
        test_data.to_csv(test_set_save_dir / f'test_data.csv', index=False)


DATA_PATH = pathlib.Path('/cv_5_folds/zoom/encrypted_traffic/data_no_nan.csv')
DATA_SET = pd.read_csv(DATA_PATH)
SAVE_DIR = pathlib.Path('/cv_5_folds/zoom/encrypted_traffic')

if __name__ == '__main__':
    build_test_datasets(data=DATA_SET, n_test_sets=5, test_set_proportion=0.1, root_save_dir=SAVE_DIR)
