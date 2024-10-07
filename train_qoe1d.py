import os
import pandas as pd
import pathlib
import torch
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt

from qoe.configs.params import TRAIN_DATA_FILE, FEATURES, LABELS, DEVICE, OPTIMIZER, LOSS_FUNCTION, TS, TEST_DATA_FILE
from qoe.models import QoEModel1D
from qoe.utils.aux_funcs import get_arg_parser, unnormalize_results, get_errors
from qoe.utils.data_utils import get_train_val_split, QoEDataset
from qoe.utils.train_utils import run_train, run_test

matplotlib.use('Agg')
plt.style.use('ggplot')

TRAIN_DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/data.csv')

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # - Get the cv_5_folds
    train_data_df = pd.read_csv(TRAIN_DATA_FILE)
    train_data_df.loc[:, LABELS[0]] /= (train_data_df.loc[:, LABELS[0]].max() - train_data_df.loc[:, LABELS[0]].min())

    # train_data_df.loc[:, FEATURES], train_pca = run_pca(dataset_df=train_data_df.loc[:, FEATURES])
    train_pca = None

    train_df, val_df = get_train_val_split(train_data_df, validation_proportion=args.val_prop)

    # - Dataset
    train_ds = QoEDataset(
        data_df=train_df,
        feature_columns=FEATURES,
        label_columns=LABELS,
        pca=train_pca,
        remove_outliers=True
    )
    val_ds = QoEDataset(
        data_df=val_df,
        feature_columns=FEATURES,
        label_columns=LABELS,
        pca=train_pca,
        remove_outliers=False

    )

    # - Data Loader
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        drop_last=True
    )

    VAL_BATCH_SIZE = args.batch_size // 4
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE if VAL_BATCH_SIZE > 0 else 1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=True
    )

    # - Build the model
    mdl = QoEModel1D(
        n_features=len(FEATURES),
        n_labels=len(LABELS),
        n_layers=args.n_layers,
        n_units=args.n_units)
    mdl.to(DEVICE)

    # - Optimizer
    optimizer = OPTIMIZER(mdl.parameters(), lr=args.lr)

    # - Loss
    loss_func = LOSS_FUNCTION

    # - Train
    # - Create the train directory
    train_save_dir = args.output_dir / f'{args.n_layers}_layers_{args.n_units}_units_{args.epochs}_epochs_{TS}'
    os.makedirs(train_save_dir)
    train_losses, val_losses = run_train(
        model=mdl,
        epochs=args.epochs,
        train_data_loader=train_dl,
        val_data_loader=val_dl,
        loss_function=loss_func,
        optimizer=optimizer,
        lr_reduce_frequency=args.lr_reduction_freq,
        lr_reduce_factor=args.lr_reduction_fctr,
        dropout_epoch_start=args.dropout_start,
        p_dropout_init=args.dropout_p,
        device=DEVICE
    )

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.suptitle('Train / Validation Loss Plot')
    plt.legend()
    plt.savefig(train_save_dir / 'train_val_loss.png')
    plt.close()

    test_data_df = pd.read_csv(TEST_DATA_FILE)
    test_ds = QoEDataset(
        data_df=test_data_df,
        feature_columns=FEATURES,
        label_columns=LABELS,
        pca=train_pca,
        remove_outliers=False
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=True
    )

    test_res = run_test(model=mdl, data_loader=test_dl, device=DEVICE)
    test_res = unnormalize_results(results=test_res, data_set=test_ds, n_columns=len(test_res.columns) // 2)
    test_res.to_csv(train_save_dir / f'test_results_{args.n_layers}_layers_{args.n_units}_units_{args.epochs}_epochs.csv')

    test_errs = get_errors(results=test_res, columns=test_ds.label_columns)
    test_errs.to_csv(train_save_dir / f'test_errors_{args.n_layers}_layers_{args.n_units}_units_{args.epochs}_epochs.csv')

    test_res = pd.concat([test_res, test_errs], axis=1)

    test_res = pd.concat([test_res, test_res]).reset_index(drop=True)

    print(f'''
===========================================================
=================== Final Stats ===========================
===========================================================
Configuration:
    > {args.n_layers} layers
    > {args.n_units} units
    > {args.epochs} epochs
    > Optimizer = {OPTIMIZER}
    > LR = {args.lr}
Mean Errors:
{test_errs.mean()}
===========================================================
    ''')
