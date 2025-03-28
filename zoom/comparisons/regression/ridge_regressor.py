import datetime
import pathlib

import pandas as pd
from sklearn.linear_model import Ridge

from regression_utils import run_cv, normalize_columns

REGRESSOR = Ridge
MODEL = 'RidgeRegression'
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/data/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/projects/phd/qoe/Code/qoe/comparisons/{MODEL}')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# NIQE, FPS, R
FEATURES = ['BW']                                                  # 3.2567+/-2.62173, 10.7236+/-11.39385, 14.2101+/-12.81807
# FEATURES = ['BW', 'PPS']                                           # 3.1388+/-3.22538, 10.7719+/-11.25124, 13.9074+/-12.56884
# FEATURES = ['BW', 'ATP']                                           # 3.2567+/-2.62173, 10.7236+/-11.39387, 14.2101+/-12.81809
# FEATURES = ['BW', 'PL']                                            # 2.7733+/-2.11633, 10.4817+/-11.72685, 11.8001+/-13.062966
# FEATURES = ['BW', 'PPS', 'ATP']                                    # 3.1388+/-3.22543, 10.7718+/-11.25124, 13.9073+/-12.56884
# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.7733+/-2.11631, 10.4816+/-11.72688, 11.8000+/-13.06290
# FEATURES = ['BW', 'PPS', 'PL']                                     # 2.7750+/-2.09047, 10.5827+/-12.43888, 11.9160+/-15.64502
# FEATURES = ['BW', 'PPS', 'ATP', 'PL']                              # 2.7750+/-2.09045, 10.5826+/-12.43876, 11.9159+/-15.64448

# FEATURES = ['L']                                                   # 5.4374+/-3.69169, 14.3924+/-10.37985, 15.9555+/-12.39015
# FEATURES = ['J']                                                   # 4.8987+/-3.85558, 14.0053+/-11.42030, 15.7201+/-13.45017
# FEATURES = ['L', 'J']                                              # 4.9053+/-3.86254, 13.9475+/-10.96043, 15.5450+/-13.22852

# FEATURES = ['PPS']                                                 # 4.0275+/-3.04774, 11.6468+/-11.87352, 15.0417+/-13.53649
# FEATURES = ['ATP']                                                 # 6.2441+/-3.41582, 16.3710+/-10.64946, 17.9436+/-12.82559
# FEATURES = ['PPS', 'ATP']                                          # 4.0274+/-3.04766, 11.6467+/-11.87332, 15.0414+/-13.53636

# FEATURES = ['BW', 'ATP', 'PL']                                     # 2.7733+/-2.11631, 10.4816+/-11.72688, 11.8000+/-13.06290
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']                    # 2.6922+/-2.12382, 10.2266+/-11.46852, 11.4663+/-14.10868
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP']              # 2.6936+/-2.12583, 10.2743+/-11.48933, 11.4195+/-14.26615
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'SP']              # 2.6995+/-2.13067, 10.3085+/-11.46215, 11.5339+/-14.42986
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'IS']              # 2.7001+/-2.14959, 10.1128+/-11.65596, 11.8034+/-16.95278
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP']        # 2.6878+/-2.10969, 10.3334+/-11.46129, 11.6597+/-14.51691
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 2.7037+/-2.12469, 10.1945+/-11.67714, 12.0344+/-18.34865

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
