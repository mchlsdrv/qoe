import os
import pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt

from configs.params import TRAIN_DATA_FILE, FEATURES, LABELS, DEVICE, OPTIMIZER, LOSS_FUNCTION, TEST_DATA_FILE, TS, OUTPUT_DIR, LAYER_ACTIVATION, AUTO_ENCODER_CODE_LENGTH_PROPORTION
from utils.aux_funcs import get_arg_parser, unnormalize_results, get_errors
from utils.data_utils import get_train_val_split, QoEDataset
from utils.train_utils import run_train, run_test

matplotlib.use('Agg')
plt.style.use('ggplot')


class AutoEncoder(torch.nn.Module):
    class Encoder(torch.nn.Module):
        def __init__(self, in_units, code_length, layer_activation):
            super().__init__()
            self._in_units = in_units
            self._code_length = code_length
            self.layers = torch.nn.ModuleList()
            self.layer_activation = layer_activation
            self._build()

        def _build(self):
            in_units = self._in_units
            out_units = in_units

            # - Encoder
            while out_units > self._code_length:
                # - Add a layer
                self._add_layer(
                    n_in=in_units,
                    n_out=out_units // 2,
                )
                out_units //= 2
                in_units = out_units

        def _add_layer(self, n_in, n_out):
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(n_in, n_out),
                    torch.nn.BatchNorm1d(n_out),
                    self.layer_activation()
                )
            )

        def forward(self, x, max_p_drop: float = 0.0):
            for lyr in self.layers:
                x = lyr(x)
                p_drop = np.random.random() * max_p_drop
                x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)
            return x

        def save_weights(self, filename: str or pathlib.Path, verbose: bool = False):
            if verbose:
                print(f'=> (ENCODER) Saving checkpoint to \'{filename}\' ...')
            torch.save(self.state_dict(), filename)

        def load_weights(self, weights_file, verbose: bool = False):
            if verbose:
                print('=> (ENCODER) Loading checkpoint ...')
            self.load_state_dict(torch.load(weights_file, weights_only=True))

        @property
        def code_length(self):
            return self._code_length

        @code_length.setter
        def code_length(self, value):
            self._code_length = value

        @property
        def in_units(self):
            return self._in_units

        @in_units.setter
        def in_units(self, value):
            self._in_units = value

    class Decoder(torch.nn.Module):
        def __init__(self, code_length, out_units, layer_activation):
            super().__init__()
            self._code_length = code_length
            self._out_units = out_units
            self.layer_activation = layer_activation
            self.layers = torch.nn.ModuleList()
            self._build()

        def _build(self):
            code_length = self._code_length
            in_units = code_length

            while code_length < self._out_units:
                # - Add a layer
                self._add_layer(
                    n_in=in_units,
                    n_out=code_length * 2,
                )
                code_length *= 2
                in_units = code_length

        def _add_layer(self, n_in, n_out):
            self.layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(n_in, n_out),
                    torch.nn.BatchNorm1d(n_out),
                    self.layer_activation()
                )
            )

        def forward(self, x, max_p_drop: float = 0.0):
            for lyr in self.layers:
                x = lyr(x)
                p_drop = np.random.random() * max_p_drop
                x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)
            return x

        def save_weights(self, filename: str or pathlib.Path, verbose: bool = False):
            if verbose:
                print(f'=> (DECODER) Saving weights to \'{filename}\' ...')
            torch.save(self.state_dict(), filename)

        def load_weights(self, weights_file, verbose: bool = False):
            if verbose:
                print('=> (DECODER) Loading weights ...')
            self.load_state_dict(torch.load(weights_file, weights_only=True))

        @property
        def out_units(self):
            return self._out_units

        @out_units.setter
        def out_units(self, value):
            self._out_units = value

        @property
        def code_length(self):
            return self._code_length

        @code_length.setter
        def code_length(self, value):
            self._code_length = value

    def __init__(self, n_features, code_length, layer_activation, reverse: bool = False, save_dir: str or pathlib.Path = pathlib.Path('./outputs')):
        super().__init__()
        self.n_features = n_features
        self.code_length = code_length
        self.layer_activation = layer_activation
        self.reverse = reverse
        self.encoder = None
        self.decoder = None

        # - Make sure the save_dir exists and of type pathlib.Path
        assert isinstance(save_dir, str) or isinstance(save_dir, pathlib.Path), f'AE: save_dir must be of type str or pathlib.Path, but is of type {type(save_dir)}!'
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_dir = pathlib.Path(self.save_dir)

        self._build()

    def _build(self):
        if self.reverse:
            self.decoder = self.Decoder(
                code_length=self.n_features,
                out_units=self.code_length,
                layer_activation=self.layer_activation
            )
            self.encoder = self.Encoder(
                in_units=self.decoder.out_units,
                code_length=self.n_features,
                layer_activation=self.layer_activation
            )
        else:
            self.encoder = self.Encoder(
                in_units=self.n_features,
                code_length=self.code_length,
                layer_activation=self.layer_activation
            )
            self.decoder = self.Decoder(
                code_length=self.encoder.code_length,
                out_units=self.n_features,
                layer_activation=self.layer_activation
            )

    def forward(self, x, p_drop: float = 0.0):
        if self.reverse:
            x = self.decoder(x, max_p_drop=p_drop)
            x = self.encoder(x, max_p_drop=p_drop)
        else:
            x = self.encoder(x, max_p_drop=p_drop)
            x = self.decoder(x, max_p_drop=p_drop)

        # - Save the weights of the encoder and the decoder separately
        self.encoder.save_weights(filename=self.save_dir / f'encoder_weights.pth')
        self.decoder.save_weights(filename=self.save_dir / f'decoder_weights.pth')

        return x


class QoEModel1D(torch.nn.Module):
    def __init__(self, n_features, n_labels, n_layers, n_units):
        super().__init__()
        self.n_features = n_features
        self.n_labels = n_labels
        self.n_layers = n_layers
        self.n_units = n_units
        self.model = None
        self.layers = torch.nn.ModuleList()
        self._build()

    def _add_layer(self, n_in, n_out, activation):
        self.layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(n_in, n_out),
                torch.nn.BatchNorm1d(n_out),
                activation()
            )
        )

    def _build(self):
        # - Add the input layer
        self._add_layer(n_in=self.n_features, n_out=self.n_units, activation=torch.nn.SiLU)

        for lyr in range(self.n_layers):
            self._add_layer(n_in=self.n_units, n_out=self.n_units, activation=torch.nn.SiLU)

        self._add_layer(n_in=self.n_units, n_out=self.n_labels, activation=torch.nn.ReLU)

    def forward(self, x, p_drop: float = 0.0):
        tmp_in = x
        for lyr in self.layers:

            x = lyr(x)

            # - Skip connection
            if tmp_in.shape == x.shape:
                x = x + tmp_in

            tmp_in = x

        x = torch.nn.functional.dropout(x, p=p_drop, training=self.training)

        return x


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # - Create the train directory
    train_save_dir = args.output_dir / f'{args.desc}_{len(FEATURES)}x{len(FEATURES) * AUTO_ENCODER_CODE_LENGTH_PROPORTION}_{args.epochs}_epochs_{TS}'
    os.makedirs(train_save_dir)

    # - Get the cv_5_folds
    train_data_df = pd.read_csv(TRAIN_DATA_FILE)

    # train_data_df.loc[:, FEATURES], train_pca = run_pca(dataset_df=train_data_df.loc[:, FEATURES])
    train_pca = None

    train_df, val_df = get_train_val_split(
        train_data_df,
        validation_proportion=args.val_prop
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
    val_ds = QoEDataset(
        data_df=val_df,
        feature_columns=FEATURES,
        label_columns=FEATURES,
        # label_columns=LABELS,
        normalize_labels=True,
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
    # mdl = QoEModel1D(
    #     n_features=len(FEATURES),
    #     n_labels=len(LABELS),
    #     n_layers=args.n_layers,
    #     n_units=args.n_units
    #     )
    mdl = AutoEncoder(
        n_features=len(FEATURES),
        code_length=len(FEATURES) * AUTO_ENCODER_CODE_LENGTH_PROPORTION,
        layer_activation=LAYER_ACTIVATION,
        reverse=True,
        save_dir=train_save_dir
    )
    mdl.to(DEVICE)

    # - Optimizer
    optimizer = OPTIMIZER(mdl.parameters(), lr=args.lr)

    # - Loss
    loss_func = LOSS_FUNCTION

    # - Train
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
        dropout_epoch_delta=args.dropout_delta,
        p_dropout_init=args.dropout_p,
        p_dropout_max=args.dropout_p_max,
        save_dir=train_save_dir,
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
        label_columns=FEATURES,
        # label_columns=LABELS,
        normalize_labels=True,
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
    test_res.to_csv(train_save_dir / f'test_results_{args.desc}_{len(FEATURES)}x{len(FEATURES) * AUTO_ENCODER_CODE_LENGTH_PROPORTION}_{args.epochs}_epochs.csv')

    test_errs = get_errors(results=test_res, columns=test_ds.label_columns)
    test_errs.to_csv(train_save_dir / f'test_errors_{args.desc}_{len(FEATURES)}x{len(FEATURES) * AUTO_ENCODER_CODE_LENGTH_PROPORTION}_{args.epochs}_epochs.csv')

    test_errs_mean = test_errs.mean()
    test_errs_mean.to_csv(train_save_dir / f'test_errors_mean_{args.desc}_{len(FEATURES)}x{len(FEATURES) * AUTO_ENCODER_CODE_LENGTH_PROPORTION}_{args.epochs}_epochs.csv')

    test_res = pd.concat([test_res, test_errs], axis=1)

    test_res = pd.concat([test_res, test_res]).reset_index(drop=True)

    print(f'''
===========================================================
=================== Final Stats ===========================
===========================================================
Configuration:
    > {args.desc} layers
    > {args.epochs} epochs
    > Optimizer = {OPTIMIZER}
    > LR = {args.lr}
Mean Errors:
{test_errs.mean()}
===========================================================
    ''')
