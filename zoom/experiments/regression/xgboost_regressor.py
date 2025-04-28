import datetime
import pathlib

import pandas as pd
from xgboost import XGBRegressor

from regression_utils import run_cv, normalize_columns

REGRESSOR = XGBRegressor
MODEL = 'XGBoost'
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/data/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/qoe/Code/qoe/comparisons/{MODEL}')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# NIQE, FPS, R
# FEATURES = ['BW']                                                  # 3.0586+/-2.80057, 10.8065+/-13.47199, 11.4117+/-16.22078
# FEATURES = ['BW', 'PPS']                                           # 2.7100+/-2.39775, 10.5465+/-21.24571, 10.9823+/-19.63156
# FEATURES = ['BW', 'ATP']                                           # 2.6630+/-2.45654, 9.5731+/-12.11165, 10.1099+/-16.95758
# FEATURES = ['BW', 'PL']                                            # 2.7212+/-2.37733, 10.5126+/-17.39065, 9.7579+/-15.43198
# FEATURES = ['BW', 'PPS', 'ATP']                                    # 2.6643+/-2.44082, 9.8031+/-12.65898, 9.9031+/-17.39466
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.7224+/-2.39683, 10.0719+/-13.67167, 9.7036+/-14.99246
# FEATURES = ['BW', 'PPS', 'PL']                                     # 2.6595+/-2.43455, 10.7550+/-18.15452, 9.9058+/-14.82607
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']                              # 2.6011+/-2.38600, 10.1484+/-13.15524, 9.7930+/-14.65192
#
# FEATURES = ['L']                                                   # 5.0591+/-4.03366, 14.5716+/-17.38775, 15.0656+/-15.07721
# FEATURES = ['J']                                                   # 4.4761+/-3.52282, 13.1239+/-13.39065, 15.1392+/-17.37904
# FEATURES = ['L', 'J']                                              # 3.8843+/-3.54050, 13.5906+/-17.67697, 13.8269+/-17.97502
#
# FEATURES = ['PPS']                                                 # 3.5355+/-3.19419, 13.5992+/-26.92947, 16.3917+/-26.17426
# FEATURES = ['ATP']                                                 # 3.5043+/-2.99700, 11.3936+/-13.12358, 14.6335+/-18.53077
# FEATURES = ['PPS', 'ATP']                                          # 3.6511+/-3.16520, 11.6935+/-15.31352, 16.1654+/-23.64834
#
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.7224+/-2.39683, 10.0719+/-13.671677, 9.7036+/-14.99246
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']                    # 2.3431+/-2.14318, 9.3994+/-12.02018, 8.0274+/-13.06922
FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP']              # 2.2003+/-2.04660, 8.8573+/-11.80195, 7.9090+/-12.24168
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'SP']              # 2.3348+/-2.13436, 9.1258+/-11.88862, 7.7563+/-11.95484
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'IS']              # 2.3413+/-2.13866, 8.5746+/-11.69048, 8.6398+/-13.02528
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP']        # 2.1882+/-1.96536, 8.6086+/-12.00959, 7.3934+/-11.36974
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 2.2155+/-2.02130, 8.2399+/-10.95442, 7.4127+/-11.57973

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
