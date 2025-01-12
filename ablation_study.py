import datetime
import os
import pathlib
from model_regression import run_ablation
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # - Input data root dir
    DATA_ROOT = pathlib.Path('./data/cv_10_folds')

    # - Output data root dir
    SAVE_DIR = pathlib.Path(f'./outputs/outputs_{TS}')
    os.makedirs(SAVE_DIR)

    DATA_DIRS = ['test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9']
    EPOCHS = [50]
    BATCHES = [128]
    # LAYERS = [16, 32, 64]
    LAYERS = [32]
    # LAYERS = [8, 16, 32, 64]
    UNITS = [256]
    # UNITS = [8, 16, 32, 64]
    LOSS_FUNCTIONS = [torch.nn.MSELoss]
    OPTIMIZERS = [torch.optim.Adam]
    INITIAL_LEARNING_RATES = [0.01]
    # INITIAL_LEARNING_RATES = [0.001]
    # FEATURES = ['Bandwidth', 'pps']
    # FEATURES = ['Bandwidth', 'avg time between packets']
    # FEATURES = ['Bandwidth', 'packets length']
    # FEATURES = ['Bandwidth', 'packets length', 'avg time between packets']
    # FEATURES = ['Bandwidth', 'pps', 'packets length']
    # FEATURES = ['Jitter']
    FEATURES = ['Latancy']
    # FEATURES = ['Bandwidth', 'pps', 'avg time between packets', 'packets length']
    # FEATURES = ['Bandwidth', 'pps', 'avg time between packets', 'packets length', 'Jitter', 'Latency']
    LABELS = ['NIQE']
    # LABELS = ['fps']
    # LABELS = ['NIQE', 'Resolution', 'fps']

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
