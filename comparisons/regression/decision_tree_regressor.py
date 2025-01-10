import datetime
import pathlib

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from regression_utils import run_cv


REGRESSOR = DecisionTreeRegressor
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/Projects/PhD/QoE/Data/zoom/encrypted_traffic/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/Projects/PhD/QoE/Code/qoe/comparisons/DCT')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# NIQE, FPS, R
# FEATURES = ['BW', 'PPS']  # 3.33+/-2.907, 11.19+/-18.299, 11.59+/-22.793
FEATURES = ['BW', 'ATP']  # 3.03+/-2.771, 11.78+/-20.439, 11.99+/-22.802
# FEATURES = ['BW', 'PL']  # 3.14+/-2.815, 11.30+/-20.179, 10.89+/-23.477
# FEATURES = ['BW', 'PPS', 'ATP']  # 3.05+/-2.889, 10.68+/-15.082, 12.09+/-26.437
# FEATURES = ['BW', 'ATP', 'PL']  # 3.14+/-2.815, 11.30+/-20.179, 10.89+/-23.477
# FEATURES = ['BW', 'PPS', 'PL']  # 3.14+/-2.815, 11.30+/-20.179, 10.89+/-23.477
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']  # 3.14+/-2.815, 11.30+/-20.179, 10.89+/-23.477

# FEATURES = ['L']  # 5.20+/-4.473, 16.07+/-35.003, 15.86+/-20.085
# FEATURES = ['J']  # 4.48+/-3.523, 13.12+/-13.391, 15.14+/-17.379
# FEATURES = ['L', 'J']  # 4.34+/-4.110, 13.87+/-20.041, 14.78+/-24.409

# FEATURES = ['PPS']  # 4.210+/-3.6881, 14.544+/-24.2634, 18.138+/-33.4549
# FEATURES = ['ATP']  # 4.060+/-3.6604, 13.511+/-22.4605, 17.179+/-29.0487
# FEATURES = ['PPS', 'ATP']

# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']  # 3.14+/-2.815, 11.30+/-20.179, 10.89+/-23.477

N_FOLDS = 10


if __name__ == '__main__':
    data_df = pd.read_csv(DATA_FILE)

    # - NIQE
    label = 'NIQE'
    output_dir = None

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
    output_dir = None

    run_cv(
        data_df=data_df,
        regressor=REGRESSOR,
        n_folds=N_FOLDS,
        features=FEATURES,
        label=label,
        data_dir=DATA_FILE.parent,
        output_dir=output_dir
    )
