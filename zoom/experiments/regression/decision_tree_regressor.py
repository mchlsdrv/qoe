import datetime
import pathlib

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from regression_utils import run_cv, normalize_columns

REGRESSOR = DecisionTreeRegressor
MODEL = 'DecisionTreeRegression'
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/data/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/qoe/Code/qoe/comparisons/{MODEL}')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# NIQE, FPS, R
# FEATURES = ['BW']                                                  # 3.5579+/-3.49154, 12.2088+/-19.86091, 14.9126+/-25.27756
FEATURES = ['BW', 'PPS']                                           # 3.2328+/-2.82432, 11.6027+/-20.67656, 11.6301+/-22.64420
# FEATURES = ['BW', 'ATP']                                           # 3.0235+/-2.75466, 11.6878+/-20.47767, 11.4802+/-21.29332
# FEATURES = ['BW', 'PL']                                            # 3.1900+/-2.84978, 11.6171+/-22.24232, 10.8917+/-23.51573
# FEATURES = ['BW', 'PPS', 'ATP']                                    # 3.0456+/-2.82552, 11.1572+/-18.415824, 12.2414+/-26.54194
# FEATURES = ['BW', 'ATP', 'PL']                                     # 3.0680+/-2.83110, 11.6957+/-21.59806, 11.1102+/-24.00114
# FEATURES = ['BW', 'PPS', 'PL']                                     # 3.0644+/-2.73909, 10.8372+/-16.76615, 10.6668+/-22.89355
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']                              # 3.0482+/-2.78320, 11.3294+/-19.26815, 11.5382+/-25.65867

# FEATURES = ['L']                                                   # 5.2009+/-4.47256, 16.0711+/-35.00304, 15.8568+/-20.08496
# FEATURES = ['J']                                                   # 4.4762+/-3.52302, 13.1240+/-13.39117, 15.1392+/-17.37908
# FEATURES = ['L', 'J']                                              # 4.3421+/-4.07628, 13.9299+/-20.30954, 14.9062+/-24.39123
#
# FEATURES = ['PPS']                                                 # 4.2099+/-3.68810, 14.5444+/-24.26344, 18.1377+/-33.45492
# FEATURES = ['ATP']                                                 # 4.0596+/-3.66044, 13.5114+/-22.460506, 17.1791+/-29.04871
# FEATURES = ['PPS', 'ATP']                                          # 4.1474+/-3.70013, 13.2169+/-20.85692, 17.8989+/-27.74595


# FEATURES = ['BW', 'ATP', 'PL']                                     # 3.0680+/-2.83110, 11.6957+/-21.59806, 11.1102+/-24.00114
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']                    # 2.8879+/-2.65599, 10.6965+/-20.79982, 10.2248+/-20.63585
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP']              # 2.8096+/-2.65185, 9.9775+/-15.24683, 8.8550+/-16.97149
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'SP']              # 2.8563+/-2.55805, 9.7363+/-13.42601, 9.9569+/-22.12693
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'IS']              # 2.8649+/-2.64347, 9.5963+/-18.25474, 10.5924+/-20.55321
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP']        # 2.7155+/-2.54444, 9.8688+/-18.43509, 8.8972+/-18.68053
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 2.8333+/-2.69492, 9.7279+/-17.00374, 10.0377+/-22.30900

N_FOLDS = 10



if __name__ == '__main__':
    data_df = pd.read_csv(DATA_FILE)
    data_df, _, _ = normalize_columns(data_df=data_df, columns=FEATURES)

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
