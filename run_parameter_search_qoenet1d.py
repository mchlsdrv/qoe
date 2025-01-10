import os
import datetime
import pathlib

import numpy as np

from utils.train_utils import search_parameters
import torch

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CV_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/rbm')
DATA_ROOT = CV_ROOT / 'niqe_rbm_cv_10_folds_int'
SAVE_DIR = CV_ROOT / f'outputs_{TS}'

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

FEATURES = ['BW', 'ATP', 'PL']  # NIQE best
# FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J']  # 3.14+/-2.815, 11.30+/-20.179, 10.89+/-23.477

DATA_DIRS = ['test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9']
EPOCHS = [50]
BATCHES = [64]
LAYERS = [64]
UNITS = [16]
LOSS_FUNCTIONS = [torch.nn.MSELoss]
OPTIMIZERS = [torch.optim.Adam]
INITIAL_LEARNING_RATES = [1e-3]
# FEATURES = [str(feat) for feat in np.arange(128)]
LABELS = ['R']


if __name__ == '__main__':
    os.makedirs(SAVE_DIR)
    search_parameters(
        test_data_root=DATA_ROOT,
        data_dirs=DATA_DIRS,
        features=FEATURES,
        labels=LABELS,
        batch_size_numbers=BATCHES,
        epoch_numbers=EPOCHS,
        layer_numbers=LAYERS,
        unit_numbers=UNITS,
        loss_functions=LOSS_FUNCTIONS,
        optimizers=OPTIMIZERS,
        initial_learning_rates=INITIAL_LEARNING_RATES,
        save_dir=SAVE_DIR
    )
