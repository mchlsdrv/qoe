import warnings
import numpy as np
import pandas as pd
import sklearn.metrics
import torchvision
from sklearn.linear_model import LogisticRegression
import torch.utils.data
from tqdm import tqdm

from qoe.configs.params import (
    DEVICE,
    DATA_FOLDER,
    BATCH_SIZE,
    RBM_VISIBLE_UNITS,
    RBM_HIDDEN_UNITS,
    RBM_K_GIBBS_STEPS,
    LR,
    MOMENTUM,
    WEIGHT_DECAY,
    EPOCHS
)

warnings.filterwarnings('ignore')


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


if __name__ == '__main__':
    train_ds = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE)

    test_ds = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False, transform=torchvision.transforms.ToTensor(), download=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

    print('\nWith RBM:')
    print('> Training... ')

    mdl = RBM(
        n_visible_units=RBM_VISIBLE_UNITS,
        n_hidden_units=RBM_HIDDEN_UNITS,
        k_gibbs_steps=RBM_K_GIBBS_STEPS,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE
    )

    mdl.fit(
        train_ds=train_ds,
        test_ds=test_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    print('**************')
    print('Without RBM:')
    print('**************')
    clf = LogisticRegression()
    train_feats_orig, train_lbls_orig = train_ds.data.view((-1, 28 * 28)), train_ds.train_labels

    print('> Training ...')
    clf.fit(train_feats_orig, train_lbls_orig)
    test_feats_orig, test_lbls_orig = test_ds.data.view((-1, 28 * 28)).numpy(), test_ds.test_labels.numpy()

    print('> Classifying ...')
    preds = clf.predict(test_feats_orig)

    n_correct, n_total = np.sum(preds == test_lbls_orig), len(test_lbls_orig)
    print(f'''
=================================================================
 Results: {n_correct} / {n_total} ({100 * n_correct / n_total:.2f}% success)
=================================================================
            ''')
