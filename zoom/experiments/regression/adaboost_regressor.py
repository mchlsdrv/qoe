import datetime
import pathlib

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor

from regression_utils import run_cv


REGRESSOR = AdaBoostRegressor
MODEL = 'AdaBoost'
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/data/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/qoe/Code/qoe/comparisons/{MODEL}')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# NIQE, FPS, R
FEATURES = ['BW']                                                  # 2.9940+/-2.39290, 10.6015+/-11.44085, 12.1790+/-13.03738
# FEATURES = ['BW', 'PPS']                                           # 2.7528+/-2.30347, 10.2950+/-10.85000, 12.1167+/-12.31608
# FEATURES = ['BW', 'ATP']                                           # 2.7850+/-2.25515, 9.9916+/-10.78727, 12.1377+/-12.43951
# FEATURES = ['BW', 'PL']                                            # 2.5214+/-2.10974, 9.4010+/-11.37592, 10.2407+/-12.98253
# FEATURES = ['BW', 'PPS', 'ATP']                                    # 2.8278+/-2.27969, 9.9402+/-10.51325, 12.7107+/-12.36280
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.5031+/-2.14695, 9.3800+/-11.20927, 10.3740+/-12.79647
# FEATURES = ['BW', 'PPS', 'PL']                                     # 2.5289+/-2.15686, 9.3793+/-11.38676, 10.1900+/-12.82334
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']                              # 2.5333+/-2.12933, 9.4612+/-11.48706, 10.0157+/-12.63148
#
# FEATURES = ['L']                                                   # 5.0347+/-3.12769, 15.4819+/-11.20592, 17.1243+/-12.16578
# FEATURES = ['J']                                                   # 4.3435+/-3.83371, 15.3222+/-13.05084, 17.2270+/-14.15640
# FEATURES = ['L', 'J']                                              # 4.1818+/-2.78732, 14.1838+/-12.95667, 16.2640+/-12.54448
#
# FEATURES = ['PPS']                                                 # 3.5341+/-2.75715, 12.6700+/-14.53222, 17.0896+/-17.20614
# FEATURES = ['ATP']                                                 # 3.3481+/-2.67414, 11.7147+/-11.71037, 16.3486+/-13.56522
# FEATURES = ['PPS', 'ATP']                                          # 3.3623+/-2.69057, 12.2512+/-11.52734, 16.0407+/-13.73501
#
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.5031+/-2.14695, 9.3800+/-11.20927, 10.3740+/-12.79647
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']                    # 2.4532+/-1.99499, 9.7478+/-11.26215, 10.0342+/-11.82088
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP']              # 2.4561+/-1.95626, 9.7344+/-10.70844, 10.5341+/-12.29174
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'SP']              # 2.4635+/-1.97513, 10.2116+/-10.40117, 10.1819+/-12.01260
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'IS']              # 2.4889+/-2.07200, 10.6375+/-10.40367, 10.4213+/-12.26260
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP']        # 2.4441+/-1.95733, 9.9662+/-10.64419, 9.9537+/-11.30885
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 2.4766+/-2.02200, 9.8532+/-10.39533, 11.3284+/-10.92740

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
