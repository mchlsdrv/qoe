import os
import pathlib
import pandas as pd
from qoe.configs.params import (
    FEATURES,
    DEVICE,
    TS,
)
from qoe.rbm import RBM
from qoe.utils.aux_funcs import get_arg_parser
from qoe.utils.data_utils import QoEDataset


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # - Get the cv_5_folds
    train_data_df = pd.read_csv(args.train_data_file)
    test_data_df = pd.read_csv(args.test_data_file)

    train_pca = None

    # train_df, val_df = get_train_val_split(train_data_df, validation_proportion=args.val_prop)

    # - Dataset
    train_ds = QoEDataset(
        data_df=train_data_df,
        feature_columns=FEATURES,
        label_columns=[args.label],
        normalize_features=True,
        normalize_labels=False,
        pca=train_pca,
        remove_outliers=True
    )

    val_ds = QoEDataset(
        data_df=test_data_df,
        feature_columns=FEATURES,
        label_columns=[args.label],
        normalize_features=True,
        normalize_labels=False,
        pca=train_pca,
        remove_outliers=False

    )

    # - Build the RBM model
    mdl = RBM(
        n_visible_units=train_ds.feature_df.shape[-1],
        n_hidden_units=args.rbm_hidden_units,
        k_gibbs_steps=args.rbm_k_gibbs_steps,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        device=DEVICE
    )
    mdl.to(DEVICE)

    # - Fit the RBM model
    rbm_features_df = mdl.fit(
        train_ds=train_ds,
        test_ds=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    #  - Change the label from the rbm.fit() to the desired label
    rbm_features_df = rbm_features_df.rename(columns={'label': args.label})

    # - Create the output file (make sure not to run over the previous file by using the time stamp)
    output_file = pathlib.Path(args.output_dir) / f'rbm/{args.label.lower()}_data_clean_rbm_{TS}.csv'

    # - Make sure the path to the file exists
    os.makedirs(output_file.parent, exist_ok=True)

    # - Save the features
    rbm_features_df.to_csv(output_file)
