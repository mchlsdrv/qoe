import os
import datetime
import pathlib
from configs.params import VAL_PROP, PAKET_SIZE_FEATURES, PIAT_FEATURES
from qoe.whatsapp.models import TransformerForRegression
import torch

from utils.train_utils import run_cv

DATA_TYPE = 'packet_size'
# DATA_TYPE = 'piat'

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# CV_ROOT_DIR = pathlib.Path('C:\\Users\\msidorov\\Desktop\\projects\\qoe\\whatsapp\\data\\packet_size_cv_10_folds_float')
CV_ROOT_DIR = pathlib.Path('/home/projects/bagon/msidorov/projects/qoe/whatsapp/data/packet_size_cv_10_folds_float')
# SAVE_DIR = pathlib.Path(f'C:\\Users\\msidorov\\Desktop\\projects\\qoe\\whatsapp\\output\\cv_{TS}')
SAVE_DIR = pathlib.Path(f'/home/projects/bagon/msidorov/projects/qoe/whatsapp/output/cv_{TS}')
os.makedirs(SAVE_DIR, exist_ok=True)

# LABELS = ['brisque']
# LABELS = ['piqe']
LABELS = ['fps']
# LABELS = ['brisque', 'piqe', 'fps']
EPOCHS = 10
BATCH_SIZE = 128

LOSS_FUNCTIONS = torch.nn.MSELoss
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 1e-3
MODEL = TransformerForRegression


def main():
    run_cv(
        model=MODEL,
        model_params={
            'model_name': 'bert-base-uncased',
        },
        n_folds=10,
        features=PAKET_SIZE_FEATURES if DATA_TYPE == 'packet_size' else PIAT_FEATURES,
        labels=LABELS,
        cv_root_dir=CV_ROOT_DIR,
        save_dir=SAVE_DIR,
        nn_params={
            'batch_size': BATCH_SIZE,
            'val_prop': VAL_PROP,
            'epochs': EPOCHS,
            'loss_function': torch.nn.MSELoss,
            'learning_rate': LEARNING_RATE,
            'optimizer': torch.optim.Adam
        }
    )


if __name__ == '__main__':
    main()
