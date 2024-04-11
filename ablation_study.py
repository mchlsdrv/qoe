import os
import pathlib
from model import run_ablation


DATA_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/data/zoom/encrypted_traffic/ablation')
SAVE_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/data/zoom/encrypted_traffic/ablation/outputs')
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = [1, 2, 10, 25, 50]
BATCHES = [4, 8, 16, 32, 64]
LAYERS = [4, 8, 16, 32, 64]
UNITS = [4, 8, 16, 32, 64]
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
        root_save_dir=SAVE_DIR
    )
