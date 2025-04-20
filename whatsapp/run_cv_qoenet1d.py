import os
import datetime
import pathlib
from configs.params import VAL_PROP
from models import QoENet1D
from utils.train_utils import run_cv
import torch


DATA_TYPE = 'packet_size'
# DATA_TYPE = 'piat'

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

CV_ROOT_DIR = pathlib.Path('/home/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/data/packet_size_cv_10_folds_float')
# CV_ROOT_DIR = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\qoe\\whatsapp\\data\\packet_size_cv_10_folds_float')
OUTPUT_DIR = pathlib.Path(f'/home/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/cv_{TS}')
# OUTPUT_DIR = pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\qoe\\whatsapp\\output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAKET_SIZE_FEATURES = [
    'number_of_packet_sizes_in_time_window',
    'number_of_unique_packet_sizes_in_time_window',
    'min_packet_size',
    'max_packet_size',
    'mean_packet_size',
    'std_packet_size',
    'q1_packet_size',
    'q2_packet_size',
    'q3_packet_size',
]

PIAT_FEATURES = [
    'number_of_piats_in_time_window',
    'number_of_unique_piats_in_time_window',
    'min_piat',
    'max_piat',
    'mean_piat',
    'std_piat',
    'q1_piat',
    'q2_piat',
    'q3_piat',
]
# LABELS = ['brisque']
# LABELS = ['piqe']
LABELS = ['fps']
# LABELS = ['brisque', 'piqe', 'fps']
EPOCHS = 500
BATCH_SIZE = 254
INPUT_SIZE = 9
OUTPUT_SIZE = 1
N_LAYERS = 32
N_UNITS = 512
LOSS_FUNCTIONS = torch.nn.MSELoss
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 1e-3
MODEL = QoENet1D


if __name__ == '__main__':
    run_cv(
        model=MODEL,
        model_params={
            'name': 'QoENet1d',
            'input_size': INPUT_SIZE,
            'output_size': OUTPUT_SIZE,
            'n_units': N_UNITS,
            'n_layers': N_LAYERS
        },
        n_folds=10,
        features=PAKET_SIZE_FEATURES if DATA_TYPE == 'packet_size' else PIAT_FEATURES,
        labels=LABELS,
        cv_root_dir=CV_ROOT_DIR,
        output_dir=OUTPUT_DIR,
        nn_params=dict(
            batch_size=BATCH_SIZE,
            val_prop=VAL_PROP,
            n_layers=N_LAYERS,
            n_units=N_LAYERS,
            epochs=EPOCHS,
            loss_function=torch.nn.MSELoss,
            learning_rate=LEARNING_RATE,
            optimizer=torch.optim.Adam
        )
    )
