import datetime
import pathlib

import pandas as pd
from xgboost import XGBRegressor

from regression_utils import run_cv


REGRESSOR = XGBRegressor
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/Projects/PhD/QoE/Data/zoom/encrypted_traffic/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/Projects/PhD/QoE/Code/qoe/comparisons/DCT')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']  # NIQE best
# FEATURES = ['BW', 'ATP', 'J']  # FPS best
FEATURES = ['BW', 'ATP', 'PL', 'L', 'J']  # R best

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
