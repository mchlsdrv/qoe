import datetime
import os
import pathlib
from model import run_ablation
import torch


TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/data/zoom/encrypted_traffic/ablation')
SAVE_DIR = pathlib.Path(f'/Users/mchlsdrv/Desktop/QoE/data/zoom/encrypted_traffic/ablation_{TS}/outputs')
os.makedirs(SAVE_DIR)

EPOCHS = [10, 50, 100]
BATCHES = [16, 32, 64]
LAYERS = [8, 16, 32, 64]
UNITS = [8, 16, 32, 64]
LOSS_FUNCTIONS = [torch.nn.MSELoss]
OPTIMIZERS = [torch.optim.SGD, torch.optim.Adam, torch.optim.Adamax]
INITIAL_LEARNING_RATES = [0.001, 0.002, 0.005, 0.008, 0.01]
FEATURES = ['Bandwidth', 'pps', 'avg time between packets', 'packets length']
LABELS = ['NIQE', 'Resolution', 'fps']

if __name__ == '__main__':
    run_ablation(
        data_root=DATA_ROOT,
        features=FEATURES,
        labels=LABELS,
        batch_size_numbers=BATCHES,
        epoch_numbers=EPOCHS,
        layer_numbers=LAYERS,
        unit_numbers=UNITS,
        loss_functions=LOSS_FUNCTIONS,
        optimizers=OPTIMIZERS,
        initial_learning_rates=INITIAL_LEARNING_RATES,
        root_save_dir=SAVE_DIR
    )
