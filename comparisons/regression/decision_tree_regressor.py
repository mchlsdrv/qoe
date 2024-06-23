import datetime
import pathlib

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from qoe.regression_utils import run_cv


REGRESSOR = DecisionTreeRegressor
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path('/qoe/comparisons/DCT')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'J', 'L']
CLASS_THRESHOLDS = [800.]
N_FOLDS = 10


if __name__ == '__main__':
    data_df = pd.read_csv(DATA_FILE)

    # - NIQE
    label = 'NIQE'
    output_dir = None
    # os.makedirs(output_dir, exist_ok=True)

    run_cv(
        data_df=data_df,
        regressor=REGRESSOR,
        n_folds=N_FOLDS,
        features=FEATURES,
        label=label,
        data_dir=DATA_FILE.parent,
        output_dir=output_dir
    )

    # - FPS
    label = 'FPS'
    # output_dir = ROOT_OUTPUT_DIR / f'outputs_{N_FOLDS}_cv_{label}_{TS}'
    output_dir = None
    # os.makedirs(output_dir, exist_ok=True)

    run_cv(
        data_df=data_df,
        regressor=REGRESSOR,
        n_folds=N_FOLDS,
        features=FEATURES,
        label=label,
        data_dir=DATA_FILE.parent,
        output_dir=output_dir
    )
    # - R
    label = 'R'
    # output_dir = ROOT_OUTPUT_DIR / f'outputs_{N_FOLDS}_cv_{label}_{TS}'
    output_dir = None
    # os.makedirs(output_dir, exist_ok=True)

    run_cv(
        data_df=data_df,
        regressor=REGRESSOR,
        n_folds=N_FOLDS,
        features=FEATURES,
        label=label,
        data_dir=DATA_FILE.parent,
        output_dir=output_dir
    )
