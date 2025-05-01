import os
import itertools
import datetime
import time

import torch

from utils.train_utils import run_cv
from configs.params import PACKET_SIZE_FEATURES, PIAT_FEATURES, MODELS, OUTPUT_DIR, EXPERIMENTS_DIR, VAL_PROP

TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LABELS = ['fps', 'brisque', 'piqe']
FEATURES = [('piat', PIAT_FEATURES), ('packet_size', PACKET_SIZE_FEATURES)]

N_CV_FOLDS = 10

MODEL_NAME = 'QoENet1D'

EPOCHS = 500
BATCH_SIZE = 254
INPUT_SIZE = 9
OUTPUT_SIZE = 1
N_LAYERS = 32
N_UNITS = 512
LOSS_FUNCTION = torch.nn.HuberLoss
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 1e-3


def main():
    for (lbl, (feat_typ, feats)) in itertools.product(LABELS, FEATURES):
        print(f'> Running {N_CV_FOLDS}-fold CV for {feat_typ.upper()} feature type and {lbl.upper()} label ...')
        cv_root_dir = OUTPUT_DIR / f'{feat_typ}_cv_10_folds'

        save_dir = EXPERIMENTS_DIR / MODEL_NAME / f'cv_{N_CV_FOLDS}_folds_{TS}/{feat_typ.lower()}_features/{lbl}_prediction/'
        os.makedirs(save_dir, exist_ok=True)

        t_start = time.time()
        with (save_dir / 'log.txt').open(mode='a') as log_fl:
            run_cv(
                model=MODELS.get(MODEL_NAME),
                model_name=MODEL_NAME,
                model_params={
                    'model_name': 'QoENet1d',
                    'input_size': INPUT_SIZE,
                    'output_size': OUTPUT_SIZE,
                    'n_units': N_UNITS,
                    'n_layers': N_LAYERS
                },
                n_folds=N_CV_FOLDS,
                features=feats,
                label=lbl,
                cv_root_dir=cv_root_dir,
                save_dir=save_dir,
                nn_params={
                    'batch_size': BATCH_SIZE,
                    'val_prop': VAL_PROP,
                    'epochs': EPOCHS,
                    'loss_function': LOSS_FUNCTION,
                    'learning_rate': LEARNING_RATE,
                    'optimizer': OPTIMIZER
                },
                log_file=log_fl
            )
            t_end = time.time() - t_start
            print(f'> The CV took total of {datetime.timedelta(seconds=t_end)}', file=log_fl)
        print(f'> The CV took total of {datetime.timedelta(seconds=t_end)}')


if __name__ == '__main__':
    main()
