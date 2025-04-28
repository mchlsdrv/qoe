import os
import pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt

from configs.params import TRAIN_DATA_FILE, FEATURES, LABELS, DEVICE, OPTIMIZER, LOSS_FUNCTION, TEST_DATA_FILE, TS, OUTPUT_DIR, LAYER_ACTIVATION, AUTO_ENCODER_CODE_LENGTH_PROPORTION
from utils.aux_funcs import get_arg_parser, unstandardize_results, get_errors
from utils.data_utils import get_train_val_split, QoEDataset
from utils.train_utils import run_train, run_test

matplotlib.use('Agg')
plt.style.use('ggplot')


class RBM(torch.nn.Module):
    def __init__(self, n_visible_units: int, n_hidden_units: int, k_gibbs_steps: int, lr: float, momentum: float, weight_decay: float, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.n_visible_units = n_visible_units
        self.n_hidden_units = n_hidden_units
        self.k_gibbs_steps = k_gibbs_steps
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device

        self.weights = None
        self.visible_bias = None
        self.hidden_bias = None

        self.weights_momentum = None
        self.visible_bias_momentum = None
        self.hidden_bias_momentum = None

        self.build()

    def build(self):
        self.weights = torch.randn(self.n_visible_units, self.n_hidden_units) * 0.1
        self.weights.to(self.device)

        self.visible_bias = torch.ones(self.n_visible_units) * 0.5
        self.visible_bias.to(self.device)

        self.hidden_bias = torch.zeros(self.n_hidden_units)
        self.hidden_bias.to(self.device)

        self.weights_momentum = torch.zeros(self.n_visible_units, self.n_hidden_units)
        self.weights_momentum.to(self.device)

        self.visible_bias_momentum = torch.zeros(self.n_visible_units)
        self.visible_bias_momentum.to(self.device)

        self.hidden_bias_momentum = torch.zeros(self.n_hidden_units)
        self.hidden_bias_momentum.to(self.device)

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = torch.nn.functional.sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.T) + self.visible_bias
        visible_probabilities = torch.nn.functional.sigmoid(visible_activations)
        return visible_probabilities

    def _get_random_probabilities(self, length):
        rand_probs = torch.rand(length)
        rand_probs.to(self.device)
        return rand_probs

    def _opt_step(self, inputs, positive_associations, negative_associations, negative_visible_probabilities, positive_hidden_probabilities, negative_hidden_probabilities):
        # - Parameter update
        self.weights_momentum *= self.momentum
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum
        self.visible_bias_momentum += torch.sum(inputs - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = inputs.size(0)

        self.weights += self.weights_momentum * self.lr / batch_size
        self.visible_bias += self.visible_bias_momentum * self.lr / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.lr / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

    def calc_contrastive_divergence(self, inputs):
        # - Positive phase
        pos_hidden_probs = self.sample_hidden(inputs)
        pos_hidden_acts = torch.as_tensor(pos_hidden_probs >= self._get_random_probabilities(self.n_hidden_units)).float()
        pos_associations = torch.matmul(inputs.T, pos_hidden_acts)

        # - Negative phase
        hidden_acts = pos_hidden_acts

        visible_probs = self.sample_visible(hidden_acts)
        hidden_probs = self.sample_hidden(visible_probs)
        hidden_acts = torch.as_tensor(hidden_probs >= self._get_random_probabilities(self.n_hidden_units)).float()
        for step in range(self.k_gibbs_steps - 1):
            visible_probs = self.sample_visible(hidden_acts)
            hidden_probs = self.sample_hidden(visible_probs)
            hidden_acts = torch.as_tensor(hidden_probs >= self._get_random_probabilities(self.n_hidden_units)).float()

        neg_visible_probs = visible_probs
        neg_hidden_probs = hidden_probs
        neg_associations = torch.matmul(neg_visible_probs.T, neg_hidden_probs)

        # - Error computation
        error = torch.sum((inputs - neg_visible_probs)**2)

        return error, pos_associations, neg_associations, neg_visible_probs, pos_hidden_probs, neg_hidden_probs

    def fit(self, train_ds: torch.utils.data.Dataset, test_ds: torch.utils.data.Dataset, epochs: int, batch_size: int):
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
        train_feats = np.zeros((len(train_ds), self.n_hidden_units))
        train_lbls = np.zeros(len(train_ds))

        for epch in tqdm(range(epochs)):
            epch_err = 0.0

            for btch, _ in train_dl:
                btch = btch.view(len(btch), self.n_visible_units)

                btch_err, pos_associations, neg_associations, neg_visible_probs, pos_hidden_probs, neg_hidden_probs = self.calc_contrastive_divergence(btch)

                # - Optimization step
                self._opt_step(
                    inputs=btch,
                    positive_associations=pos_associations,
                    negative_associations=neg_associations,
                    negative_visible_probabilities=neg_visible_probs,
                    positive_hidden_probabilities=pos_hidden_probs,
                    negative_hidden_probabilities=neg_hidden_probs,
                )

                epch_err += btch_err

            print(f'\nEpoch #{epch} Error: {epch_err:.3f}')

            print('> Extracting features ...')
            train_feats = np.zeros((len(train_ds), self.n_hidden_units))
            train_lbls = np.zeros(len(train_ds))
            test_feats = np.zeros((len(test_ds), self.n_hidden_units))
            test_lbls = np.zeros(len(test_ds))

            btch: torch.Tensor
            for idx, (btch, lbls) in enumerate(train_dl):
                btch = btch.view(len(btch), self.n_visible_units)  # flatten input data

                btch = btch.to(DEVICE)

                train_feats[idx * batch_size:idx * batch_size + len(btch)] = self.sample_hidden(btch).cpu().numpy()
                train_lbls[idx * batch_size:idx * batch_size + len(btch)] = lbls.numpy().flatten()

            for idx, (btch, lbls) in enumerate(test_dl):
                btch = btch.view(len(btch), self.n_visible_units)  # flatten input data

                btch = btch.to(DEVICE)

                test_feats[idx * batch_size:idx * batch_size + len(btch)] = self.sample_hidden(btch).cpu().numpy()
                test_lbls[idx * batch_size:idx * batch_size + len(btch)] = lbls.numpy().flatten()

            # mse = sklearn.metrics.mean_squared_error(train_fea)
            print('> Classifying ...')
            clf = LogisticRegression()
            clf.fit(train_feats, np.floor(train_lbls))
            preds = clf.predict(test_feats)

            n_correct, n_total = np.sum(preds == np.floor(test_lbls)), len(test_lbls)
            print(f'''
=================================================================
 Results: {n_correct} / {n_total} ({100 * n_correct / n_total:.2f}% success)
=================================================================
        ''')

        data_df = pd.DataFrame(train_feats)
        data_df['label'] = train_lbls

        return data_df
        
        
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


class QoENet1D(torch.nn.Module):
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

        self._add_layer(n_in=self.n_units, n_out=self.n_labels, activation=torch.nn.Tanh)
        # self._add_layer(n_in=self.n_units, n_out=self.n_labels, activation=torch.nn.ReLU)

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
    # mdl = QoENet1D(
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
    test_res = unstandardize_results(results=test_res, data_set=test_ds, n_columns=len(test_res.columns) // 2)
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
