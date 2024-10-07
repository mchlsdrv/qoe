import os
import datetime
import pathlib

import numpy as np

from utils.train_utils import run_cv
import torch

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
CV_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/rbm')
DATA_ROOT = CV_ROOT / 'niqe_rbm_cv_10_folds_int'
SAVE_DIR = CV_ROOT / f'outputs_{TS}'

DATA_DIRS = ['test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9']
EPOCHS = [50]
BATCHES = [64]
LAYERS = [64]
UNITS = [16]
LOSS_FUNCTIONS = [torch.nn.MSELoss]
OPTIMIZERS = [torch.optim.Adam]
INITIAL_LEARNING_RATES = [1e-3]
FEATURES = [str(feat) for feat in np.arange(128)]
LABELS = ['R']


if __name__ == '__main__':
    os.makedirs(SAVE_DIR)
    run_cv(
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
