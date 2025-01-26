import datetime
import os
import pathlib
from model_regression import run_ablation
import warnings
import torch

from qoe.configs.params import RANDOM_SEED, DEVICE

torch.manual_seed(RANDOM_SEED)
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Input data root dir
    DATA_ROOT = pathlib.Path('data/cv_10_folds')
    # DATA_ROOT = pathlib.Path('data/cv_10_folds_2')

    # - Output data root dir
    SAVE_DIR = pathlib.Path(f'./outputs/outputs_{TS}')
    os.makedirs(SAVE_DIR)

    # - CONSTANTS
    DATA_DIRS = ['test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9']
    BATCHES = [128]
    EPOCHS = [50]

    # - HYPERPARAMS
    LAYERS = [8]
    # LAYERS = [16]
    # LAYERS = [32]

    # UNITS = [8]
    # UNITS = [16]
    UNITS = [32]
    # UNITS = [64]
    # UNITS = [128]
    # UNITS = [256]
    # UNITS = [512]
    # UNITS = [1024]

    LOSS_FUNCTIONS = [torch.nn.MSELoss]
    OPTIMIZERS = [torch.optim.Adam]
    # INITIAL_LEARNING_RATES = [0.01]
    # INITIAL_LEARNING_RATES = [0.005]  # ~5%
    # INITIAL_LEARNING_RATES = [0.001]  # = 3.59%
    # INITIAL_LEARNING_RATES = [0.0008]  #  3.5614+/-0.42748,  14.2700+/-1.68794, 11.1842+/-2.31188
    INITIAL_LEARNING_RATES = [0.0005]  #  3.5798+/-0.40347,  14.2676+/-1.64145, 11.3202+/-2.50084

    # - Features
    # INITIAL_LEARNING_RATES = [0.001]
    # FEATURES = ['BW']
    # FEATURES = ['BW', 'PPS']
    # FEATURES = ['BW', 'ATP']
    # FEATURES = ['BW', 'PL']
    # FEATURES = ['BW', 'PPS', 'ATP']
    # FEATURES = ['BW', 'ATP', 'PL']
    # FEATURES = ['BW', 'PPS', 'PL']
    # FEATURES = ['BW', 'PPS', 'ATP', 'PL']
    # FEATURES = ['L']
    # FEATURES = ['J']
    # FEATURES = ['L', 'J']
    # FEATURES = ['Latancy', 'Jitter']
    # FEATURES = ['PPS']
    # FEATURES = ['ATP']
    # FEATURES = ['L', 'J', 'PPS', 'ATP']
    FEATURES = [ 'BW', 'L', 'J', 'PPS', 'DP', 'SP', 'ATP', 'PL', 'IS' ]
    # FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'J', 'L', ]

    # - Labels
    # LABELS = ['NIQE', 'Resolution', 'fps']
    LABELS = ['NIQE', 'R', 'FPS']

    run_ablation(
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
        save_dir=SAVE_DIR,
        device=DEVICE
    )
