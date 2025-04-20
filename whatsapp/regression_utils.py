import os

import numpy as np
import pandas as pd
import pathlib

from configs.params import EPSILON, RANDOM_SEED
from data_division import build_test_datasets
np.random.seed(RANDOM_SEED)
DEBUG = False


def calc_errors(true, predicted):
    errors = np.abs(100 - predicted.flatten() * 100 / (true.flatten() + EPSILON))

    return errors


def train_regressor(X, y, regressor):
    # - Create the model
    model = regressor()

    # - Fit the model on the train data
    model.fit(X, y.flatten())

    return model


def eval_regressor(X, y, model):
    # - Predict the train_test data
    y_pred = model.predict(X)

    errors = calc_errors(true=y, predicted=y_pred)

    return y_pred, errors
