import datetime
import os
import pathlib
from model_regression import run_ablation
import torch


if __name__ == '__main__':
    TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ABLATION_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic')
    DATA_ROOT = ABLATION_ROOT / 'cv_10_folds'
    SAVE_DIR = ABLATION_ROOT / f'outputs_{TS}'
    os.makedirs(SAVE_DIR)

    DATA_DIRS = ['test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9']
    EPOCHS = [50]
    BATCHES = [128]
    LAYERS = [16, 32, 64]
    # LAYERS = [8, 16, 32, 64]
    UNITS = [256]
    # UNITS = [8, 16, 32, 64]
    LOSS_FUNCTIONS = [torch.nn.MSELoss]
    OPTIMIZERS = [torch.optim.Adam]
    INITIAL_LEARNING_RATES = [0.001]
    FEATURES = ['Bandwidth', 'pps', 'avg time between packets', 'packets length']
    # LABELS = ['NIQE']
    LABELS = ['NIQE', 'Resolution', 'fps']

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
        save_dir=SAVE_DIR
    )
