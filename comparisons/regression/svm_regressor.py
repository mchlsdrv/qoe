import datetime
import pathlib

import pandas as pd
from sklearn import svm

from regression_utils import run_cv


REGRESSOR = svm.SVR
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/Projects/PhD/QoE/Data/zoom/encrypted_traffic/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/Projects/PhD/QoE/Code/qoe/comparisons/DCT')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# NIQE, FPS, R
# FEATURES = ['BW']                                                  # 3.2567+/-2.62173, 10.7236+/-11.39385, 14.2101+/-12.81807
# FEATURES = ['BW', 'PPS']                                           # 3.1377+/-3.19918, 10.7713+/-11.25163, 13.9074+/-12.56883
# FEATURES = ['BW', 'ATP']                                           # 3.2567+/-2.62173, 10.7236+/-11.39385, 14.2101+/-12.81807
# FEATURES = ['BW', 'PL']                                            # 2.7706+/-2.11559, 10.4799+/-11.72183, 11.8001+/-13.06268
# FEATURES = ['BW', 'PPS', 'ATP']                                    # 3.1377+/-3.19918, 10.7713+/-11.25163, 13.9074+/-12.56883
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.7706+/-2.11559, 10.4799+/-11.72183, 11.8001+/-13.06268
# FEATURES = ['BW', 'PPS', 'PL']                                     # 2.7678+/-2.09515, 10.5754+/-12.39795, 11.9153+/-15.63669
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']                              # 2.7678+/-2.09515, 10.5754+/-12.39795, 11.9153+/-15.63669
#
# FEATURES = ['L']                                                   # 5.4400+/-3.68757, 14.3927+/-10.37958, 15.9556+/-12.39014
# FEATURES = ['J']                                                   # 5.1978+/-3.62184, 14.0571+/-11.25379, 15.7285+/-13.40155
# FEATURES = ['L', 'J']                                              # 5.3514+/-3.68795, 13.9909+/-10.78614, 15.5448+/-13.15993

# FEATURES = ['PPS']                                                 # 4.0297+/-3.04602, 11.6472+/-11.87285, 15.0417+/-13.53648
# FEATURES = ['ATP']                                                 # 6.2470+/-3.41704, 16.3775+/-10.65048, 17.9478+/-12.82805
# FEATURES = ['PPS', 'ATP']                                          # 4.0297+/-3.04602, 11.6472+/-11.87285, 15.0417+/-13.53648

# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.7706+/-2.11559, 10.4799+/-11.72183, 11.8001+/-13.06268
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']                    # 2.7070+/-2.11323, 10.2011+/-11.41108, 11.4622+/-14.10149
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP']              # 2.7093+/-2.11200, 10.2447+/-11.43241, 11.4148+/-14.25958
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'SP']              # 2.7104+/-2.12194, 10.2776+/-11.40427, 11.5284+/-14.42323
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'IS']              # 2.7070+/-2.11323, 10.0707+/-11.55243, 11.7964+/-16.93786
FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP']          # 2.6957+/-2.10428, 10.3013+/-11.40323, 11.6544+/-14.50924
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 2.6957+/-2.10428, 10.1509+/-11.56672, 12.0271+/-18.33098
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
