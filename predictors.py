import os
import pathlib
import numpy as np
import pandas as pd
import torch

from models import AutoEncoder as AE

from configs.params import (
    TRAIN_DATA_FILE,
    FEATURES,
    LABELS,
    DEVICE,
    OPTIMIZER,
    LOSS_FUNCTION,
    TEST_DATA_FILE,
    TS,
    OUTPUT_DIR,
    LAYER_ACTIVATION,
    AUTO_ENCODER_CODE_LENGTH_PROPORTION
)
from utils.aux_funcs import get_arg_parser, unnormalize_results, get_errors
from utils.data_utils import get_train_val_split, QoEDataset
from utils.train_utils import run_train, run_test

ae_mdl = AE(
    n_features=len(FEATURES),
    code_length=len(FEATURES) * AUTO_ENCODER_CODE_LENGTH_PROPORTION,
    layer_activation=LAYER_ACTIVATION,
    reverse=True,
    save_dir=OUTPUT_DIR
)

ENC_WEIGHTS = pathlib.Path('/Users/mchlsdrv/Desktop/BGU/PhD/QoE/Code/qoe/outputs/auto_encoder_4x128_100_epochs_2024-10-08_01-14-25/encoder_weights.pth')
ENC_WEIGHTS.is_file()

DEC_WEIGHTS = pathlib.Path('/Users/mchlsdrv/Desktop/BGU/PhD/QoE/Code/qoe/outputs/auto_encoder_4x128_100_epochs_2024-10-08_01-14-25/decoder_weights.pth')
DEC_WEIGHTS.is_file()

ae_mdl.encoder.load_weights(ENC_WEIGHTS)
ae_mdl.decoder.load_weights(DEC_WEIGHTS)

enc = ae_mdl.encoder
dec = ae_mdl.decoder
dec


# - Get the cv_5_folds
train_data_df = pd.read_csv(TRAIN_DATA_FILE)

# train_data_df.loc[:, FEATURES], train_pca = run_pca(dataset_df=train_data_df.loc[:, FEATURES])
train_pca = None

train_df, val_df = get_train_val_split(
    train_data_df,
    validation_proportion=0.
)

# - Dataset
train_ds = QoEDataset(
    data_df=train_df,
    feature_columns=FEATURES,
    label_columns=FEATURES,
    # label_columns=LABELS,
    normalize_labels=True,
    pca=train_pca,
    remove_outliers=True
)

# - Data Loader
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)

btch = next(iter(train_dl))
btch[0]

btch_dec = dec(btch[0])
btch_rec = enc(btch_dec).detach().cpu()
np.abs(btch_rec - btch[0])
