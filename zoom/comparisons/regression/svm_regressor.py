import datetime
import pathlib

import pandas as pd
from sklearn import svm

from regression_utils import run_cv


REGRESSOR = svm.SVR
MODEL = 'SVM'
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/data/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/qoe/Code/qoe/comparisons/{MODEL}')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# NIQE, FPS, R
# FEATURES = ['BW']                                                  # 2.6680+/-2.53945, 8.9186+/-12.09082, 11.0526+/-15.99866
# FEATURES = ['BW', 'PPS']                                           # 2.6864+/-2.50884, 9.2655+/-11.89466, 11.3982+/-15.81209
# FEATURES = ['BW', 'ATP']                                           # 2.6869+/-2.50874, 9.2681+/-11.89370, 11.4008+/-15.81213
# FEATURES = ['BW', 'PL']                                            # 2.6868+/-2.50897, 9.2667+/-11.89418, 11.3994+/-15.81199
# FEATURES = ['BW', 'PPS', 'ATP']                                    # 2.6973+/-2.49024, 9.3252+/-11.87195, 11.4569+/-15.80569
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.6976+/-2.49045, 9.3261+/-11.87109, 11.4575+/-15.80526
# FEATURES = ['BW', 'PPS', 'PL']                                     # 2.6971+/-2.49014, 9.3244+/-11.87209, 11.4564+/-15.80604
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']                              # 2.7033+/-2.48600, 9.3521+/-11.86198, 11.4833+/-15.80123
#
# FEATURES = ['L']                                                   # 4.7310+/-3.85246, 12.2665+/-11.85310, 12.9956+/-17.33425
# FEATURES = ['J']                                                   # 4.3435+/-3.83371, 12.0832+/-13.26772, 12.5900+/-16.98145
# FEATURES = ['L', 'J']                                              # 4.6873+/-3.74295, 12.3270+/-11.76430, 13.0186+/-17.39604

# FEATURES = ['PPS']                                                 # 3.2766+/-2.86506, 9.9519+/-12.57345, 11.7446+/-16.11497
# FEATURES = ['ATP']                                                 # 3.1307+/-2.74764, 9.7079+/-12.44218, 11.5413+/-16.18538
# FEATURES = ['PPS', 'ATP']                                          # 3.3197+/-2.77402, 10.3689+/-12.34930, 12.2357+/-15.88880

# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.6976+/-2.49045, 9.3261+/-11.87109, 11.4575+/-15.80526
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']                    # 2.7131+/-2.48717, 9.3716+/-11.85159, 11.5076+/-15.79726
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP']              # 2.7087+/-2.49366, 9.3803+/-11.85134, 11.5158+/-15.79486
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'SP']              # 2.7182+/-2.48275, 9.3799+/-11.84621, 11.5153+/-15.79381
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'IS']              # 2.7143+/-2.48550, 9.3765+/-11.84883, 11.5146+/-15.79501
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP']        # 2.7066+/-2.48440, 9.3845+/-11.84577, 11.5221+/-15.79091
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 2.7075+/-2.48409, 9.3875+/-11.84459, 11.5263+/-15.78940

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
