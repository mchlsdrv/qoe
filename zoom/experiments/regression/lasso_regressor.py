import datetime
import pathlib

import pandas as pd
from sklearn.linear_model import Lasso

from regression_utils import run_cv, normalize_columns

REGRESSOR = Lasso
MODEL = 'LassoRegression'
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/data/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/qoe/Code/qoe/comparisons/{MODEL}')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# NIQE, FPS, R
FEATURES = ['BW']                                                  # 3.2567+/-2.62172, 10.7236+/-11.39385, 14.2101+/-12.81807
# FEATURES = ['BW', 'PPS']                                           # 3.1368+/-3.17369, 10.7707+/-11.25203, 13.9074+/-12.56882
# FEATURES = ['BW', 'ATP']                                           # 3.2567+/-2.62172, 10.7236+/-11.39385, 14.2101+/-12.81807
# FEATURES = ['BW', 'PL']                                            # 2.7690+/-2.11487, 10.4783+/-11.71700, 11.8001+/-13.06271
# FEATURES = ['BW', 'PPS', 'ATP']                                    # 3.1368+/-3.17369, 10.7707+/-11.25203, 13.9074+/-12.56882
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.7690+/-2.11487, 10.4783+/-11.71700, 11.8001+/-13.06271
# FEATURES = ['BW', 'PPS', 'PL']                                     # 2.7706+/-2.09784, 10.5687+/-12.35986, 11.9154+/-15.63761
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']                              # 2.7706+/-2.09784, 10.5687+/-12.35986, 11.9154+/-15.63761

# FEATURES = ['L']                                                   # 5.4426+/-3.68352, 14.3930+/-10.37932, 15.9556+/-12.39014
# FEATURES = ['J']                                                   # 5.6016+/-3.47172, 14.0834+/-11.19430, 15.7207+/-13.44624
# FEATURES = ['L', 'J']                                              # 5.4426+/-3.68352, 14.0339+/-10.70200, 15.5448+/-13.22132

# FEATURES = ['PPS']                                                 # 4.0319+/-3.04433, 11.6475+/-11.87220, 15.0417+/-13.53647
# FEATURES = ['ATP']                                                 # 6.2470+/-3.41704, 16.3775+/-10.65048, 17.9478+/-12.82805
# FEATURES = ['PPS', 'ATP']                                          # 4.0319+/-3.04433, 11.6475+/-11.87220, 15.0417+/-13.53647

# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.7690+/-2.11487, 10.4783+/-11.71700, 11.8001+/-13.06271
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']                    # 2.7072+/-2.12349, 10.1967+/-11.39498, 11.4624+/-14.10303
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP']              # 2.7090+/-2.12321, 10.2400+/-11.41621, 11.4152+/-14.26057
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'SP']              # 2.7108+/-2.13264, 10.2722+/-11.38737, 11.5289+/-14.42403
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'IS']              # 2.7072+/-2.12349, 10.0566+/-11.49809, 11.7993+/-16.94919
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP']        # 2.6981+/-2.11466, 10.2952+/-11.38667, 11.6549+/-14.51013
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 2.6981+/-2.11466, 10.1369+/-11.50678, 12.0306+/-18.34731

N_FOLDS = 10


if __name__ == '__main__':
    data_df = pd.read_csv(DATA_FILE)
    data_df, _, _ = normalize_columns(data_df=data_df, columns=FEATURES)

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
