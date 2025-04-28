import datetime
import pathlib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from regression_utils import run_cv, normalize_columns

REGRESSOR = RandomForestRegressor
MODEL = 'RandomForestRegression'
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/data/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/qoe/Code/qoe/comparisons/{MODEL}')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# NIQE, FPS, R
# FEATURES = ['BW']                                                  # 3.0929+/-2.96425, 10.2299+/-12.49784, 12.3180+/-17.21777
# FEATURES = ['BW', 'PPS']                                           # 2.4791+/-2.29787, 8.7597+/-11.21779, 9.0939+/-13.24784
# FEATURES = ['BW', 'ATP']                                           # 2.4821+/-2.27868, 8.6890+/-11.03613, 9.1520+/-12.84909
# FEATURES = ['BW', 'PL']                                            # 2.4618+/-2.19145, 8.8290+/-11.07290, 8.9483+/-12.87244
# FEATURES = ['BW', 'PPS', 'ATP']                                    # 2.4865+/-2.29245, 8.6234+/-10.92435, 9.0495+/-12.92580
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.4193+/-2.19271, 8.7870+/-11.27411, 8.5808+/-12.53022
# FEATURES = ['BW', 'PPS', 'PL']                                     # 2.4068+/-2.18342, 8.8588+/-11.48019, 8.6155+/-12.47232
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']                              # 2.4103+/-2.22076, 8.7971+/-11.31665, 8.5422+/-12.36409
#
# FEATURES = ['L']                                                   # 4.9084+/-4.07677, 14.0941+/-16.83355, 15.2007+/-15.82445
# FEATURES = ['J']                                                   # 4.4769+/-3.520882, 13.0156+/-12.69118, 14.9837+/-15.71454
# FEATURES = ['L', 'J']                                              # 3.7390+/-3.24772, 11.9126+/-13.56525, 12.1970+/-15.88492

# FEATURES = ['PPS']                                                 # 3.7029+/-3.28814, 13.2830+/-21.19400, 16.0137+/-25.82367
# FEATURES = ['ATP']                                                 # 3.6192+/-3.13448, 11.3991+/-13.65226, 14.7123+/-19.22653
# FEATURES = ['PPS', 'ATP']                                          # 3.5277+/-3.06866, 11.0337+/-12.55772, 14.3481+/-17.78746


# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.4193+/-2.19271, 8.7870+/-11.27411, 8.5808+/-12.53022
FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']                    # 2.0945+/-1.93787, 8.2115+/-10.53703, 7.6430+/-11.17112
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP']              # 2.0346+/-1.88813, 7.9456+/-9.97394, 7.3228+/-10.86842
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'SP']              # 2.0989+/-1.95198, 8.0212+/-9.97745, 7.5061+/-10.89929
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'IS']              # 2.0980+/-1.94857, 7.7969+/-10.06816, 7.7354+/-11.00372
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP']        # 2.0222+/-1.88477, 7.8482+/-9.81416, 7.0263+/-10.28238
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 2.0473+/-1.91525, 7.5720+/-9.62648, 7.1896+/-10.35819

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
