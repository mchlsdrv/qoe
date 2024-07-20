import torch


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
        hid_acts = pos_hidden_acts

        visible_probs = self.sample_visible(hid_acts)
        hidden_probs = self.sample_hidden(visible_probs)
        hidden_acts = torch.as_tensor(hidden_probs >= self._get_random_probabilities(self.n_hidden_units)).float()
        for step in range(self.k_gibbs_steps - 1):
            visible_probs = self.sample_visible(hid_acts)
            hidden_probs = self.sample_hidden(visible_probs)
            hidden_acts = torch.as_tensor(hidden_probs >= self._get_random_probabilities(self.n_hidden_units)).float()

        neg_visible_probs = visible_probs
        neg_hidden_probs = hidden_probs
        neg_associations = torch.matmul(neg_visible_probs.T, neg_hidden_probs)

        # - Optimization step
        self._opt_step(
            inputs=inputs,
            positive_associations=pos_associations,
            negative_associations=neg_associations,
            negative_visible_probabilities=neg_visible_probs,
            positive_hidden_probabilities=pos_hidden_probs,
            negative_hidden_probabilities=neg_hidden_probs,
        )

        # - Error computation
        error = torch.sum((input - neg_visible_probs)**2)

        return error




