import os
import pathlib
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

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
