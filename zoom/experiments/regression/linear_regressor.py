import datetime
import pathlib

import pandas as pd
from sklearn.linear_model import LinearRegression

from regression_utils import run_cv, normalize_columns

REGRESSOR = LinearRegression
MODEL = 'LinearRegression'
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/data/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/qoe/Code/qoe/comparisons/{MODEL}')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# NIQE, FPS, R
# FEATURES = ['BW']                                                  # 3.2567+/-2.62173, 10.7236+/-11.39385, 14.2101+/-12.81807
# FEATURES = ['BW', 'PPS']                                           # 3.1388+/-3.22538, 10.7719+/-11.25124, 13.9074+/-12.56884
# FEATURES = ['BW', 'ATP']                                           # 3.2958+/-2.66648, 10.9573+/-11.78094, 14.2235+/-13.29360
# FEATURES = ['BW', 'PL']                                            # 2.7733+/-2.11633, 10.4817+/-11.72685, 11.8001+/-13.06296
# FEATURES = ['BW', 'PPS', 'ATP']                                    # 3.1559+/-4.44079, 10.7882+/-11.78717, 13.3205+/-13.28052
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.6821+/-2.05053, 10.2372+/-12.36763, 11.0918+/-13.01265
# FEATURES = ['BW', 'PPS', 'PL']                                     # 2.7750+/-2.09047, 10.5827+/-12.43888, 11.9160+/-15.64504
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']                              # 2.7259+/-2.22694, 10.2366+/-12.36116, 11.1269+/-13.11758
#
# FEATURES = ['L']                                                   # 5.4374+/-3.69169, 14.3924+/-10.37985, 15.9555+/-12.39015
# FEATURES = ['J']                                                   # 4.8986+/-3.85563, 14.0053+/-11.42049, 15.7200+/-13.45032
# FEATURES = ['L', 'J']                                              # 4.9053+/-3.86260, 13.9475+/-10.96060, 15.5450+/-13.22873

# FEATURES = ['PPS']                                                 # 4.0275+/-3.04774, 11.6468+/-11.87352, 15.0417+/-13.53649
# FEATURES = ['ATP']                                                 # 3.5390+/-2.83060, 11.1801+/-11.88351, 14.1247+/-13.91587
# FEATURES = ['PPS', 'ATP']                                          # 3.5925+/-2.98194, 11.4287+/-12.10340, 13.8567+/-14.63747

# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.6821+/-2.05053, 10.2372+/-12.36763, 11.0918+/-13.01265
FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']                    # 2.6877+/-2.31728, 9.9335+/-11.44815, 10.8262+/-12.87545
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP']              # 2.6806+/-2.32592, 9.9627+/-11.45573, 10.7969+/-12.90673
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'SP']              # 2.6944+/-2.31173, 9.9692+/-11.42537, 10.8021+/-12.92279
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'IS']              # 2.7028+/-2.32837, 9.8307+/-11.21504, 11.0685+/-13.05403
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP']        # 2.6552+/-2.27299, 9.9741+/-11.42042, 10.9187+/-12.93903
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 2.6690+/-2.28152, 9.8692+/-11.17542, 11.1807+/-13.19941

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
