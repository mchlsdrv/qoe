import os

import numpy as np
import pandas as pd
import pathlib

from configs.params import EPSILON, RANDOM_SEED
from data_division import build_test_datasets
np.random.seed(RANDOM_SEED)
DEBUG = False


def calc_errors(true, predicted):
    errors = np.abs(100 - predicted.flatten() * 100 / (true.flatten() + EPSILON))

    return errors


def normalize_columns(data_df, columns):
    data_df = data_df.astype(float)
    mu, std = data_df.loc[:, columns].mean(), data_df.loc[:, columns].std()
    data_df.loc[:, columns] = (data_df.loc[:, columns] - mu) / std
    return data_df, mu, std


def train_regressor(X, y, regressor):
    # - Create the model
    model = regressor()

    # - Fit the model on the train data
    model.fit(X, y.flatten())

    return model


def eval_regressor(X, y, model):
    # - Predict the train_test data
    y_pred = model.predict(X)

    errors = calc_errors(true=y, predicted=y_pred)

    return y_pred, errors


def run_cv(data_df: pd.DataFrame, regressor, n_folds: int, features: list, label: str, data_dir: pathlib.Path, output_dir: pathlib.Path or None):
    cv_root_dir = data_dir / f'cv_{n_folds}_folds'
    if not cv_root_dir.is_dir():
        os.makedirs(cv_root_dir, exist_ok=True)
        build_test_datasets(data_df, n_folds=n_folds, root_save_dir=cv_root_dir)

    results = pd.DataFrame(columns=['true', 'predicted', 'error (%)'], dtype=np.float32)
    for cv_dir in os.listdir(cv_root_dir):
        if cv_dir[0] != '.':
            train_df = pd.read_csv(cv_root_dir / cv_dir / 'train_data.csv')
            X_train = train_df.loc[:, features].values
            y_train = train_df.loc[:, [label]].values
            model = train_regressor(X=X_train, y=y_train, regressor=regressor)

            test_df = pd.read_csv(cv_root_dir / cv_dir / 'test_data.csv')
            X_test = test_df.loc[:, features].values
            y_test = test_df.loc[:, [label]].values

            preds, errors = eval_regressor(X=X_test, y=y_test, model=model)
            results = pd.concat([
                results,
                pd.DataFrame(
                    {
                        'true': y_test.flatten(),
                        'predicted': preds.flatten(),
                        'error (%)': errors.flatten()
                    }
                )
            ], ignore_index=True)

    if isinstance(output_dir, pathlib.Path):
        results.to_csv(output_dir / 'final_results.csv')
    mean_error = results.loc[:, "error (%)"].mean()
    std_error = results.loc[:, "error (%)"].std()
    print(f'''
Mean Stats on {n_folds} CV for {label}:
    Mean Errors (%)
    ---------------
    {mean_error:.2f}+/-{std_error:.3f}
    ''')
    return results
