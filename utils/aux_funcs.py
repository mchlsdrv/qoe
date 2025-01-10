import argparse
import pathlib

import numpy as np
import pandas as pd
import sklearn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from configs.params import (
    N_LAYERS,
    N_UNITS,
    EPOCHS,
    LR,
    LR_REDUCTION_FREQ,
    LR_REDUCTION_FCTR,
    DROPOUT_START,
    DROPOUT_P,
    BATCH_SIZE,
    VAL_PROP,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    OUTPUT_DIR,
    OUTLIER_TH, LABEL, MOMENTUM, WEIGHT_DECAY, RBM_VISIBLE_UNITS, RBM_HIDDEN_UNITS, RBM_K_GIBBS_STEPS, EPSILON, DESCRIPTION, DROPOUT_DELTA, DROPOUT_P_MAX
)
from utils.data_utils import QoEDataset
plt.style.use('ggplot')


def plot_losses(train_losses, val_losses, save_dir: pathlib.Path):
    # - Plot the train / val losses
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.suptitle('Train / Validation Loss Plot')
    plt.legend()
    plt.savefig(save_dir / 'train_val_loss.png')
    plt.close()


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # - GENERAL PARAMETERS
    parser.add_argument('--n_layers', type=int, default=N_LAYERS, help='Number of NN-Blocks in the network')
    parser.add_argument('--n_units', type=int, default=N_UNITS, help='Number of units in each NN-Block')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='The number of samples in each batch')
    parser.add_argument('--lr', type=int, default=LR, help='Represents the initial learning rate of the optimizer')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--val_prop', type=float, default=VAL_PROP, help=f'The proportion of the data which will be set aside, and be used in the process of validation')
    parser.add_argument('--outlier_th', type=int, default=OUTLIER_TH, help='Represents the number of STDs from the mean to remove samples with')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='The path to the directory where the outputs will be placed')
    parser.add_argument('--lr_reduction_freq', type=int, default=LR_REDUCTION_FREQ, help='Represents the number of epochs for the LR reduction')
    parser.add_argument('--lr_reduction_fctr', type=float, default=LR_REDUCTION_FCTR, help='Represents the factor by which the LR reduced each LR_REDUCTION_FREQ epochs')
    parser.add_argument('--dropout_start', type=int, default=DROPOUT_START, help='The epoch when the dropout technique start being applied')
    parser.add_argument('--dropout_delta', type=int, default=DROPOUT_DELTA, help='The number of epochs in which the p_drop will be constant')
    parser.add_argument('--dropout_p', type=float, default=DROPOUT_P, help='The probability of the unit to be zeroed out')
    parser.add_argument('--dropout_p_max', type=float, default=DROPOUT_P_MAX, help='The maximal probability of the unit to be zeroed out')
    parser.add_argument('--train_data_file', type=str, default=TRAIN_DATA_FILE, help='The path to the train data file')
    parser.add_argument('--desc', type=str, default=DESCRIPTION, help='The description of the current experiment')
    parser.add_argument('--test_data_file', type=str, default=TEST_DATA_FILE, help='The path to the train_test data file')
    parser.add_argument('--label', type=str, default=LABEL, help='The label column name')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='The momentum value to use in training')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='The weight decay value to use in training')
    parser.add_argument('--rbm_visible_units', type=int, default=RBM_VISIBLE_UNITS, help='The number of visible units')
    parser.add_argument('--rbm_hidden_units', type=int, default=RBM_HIDDEN_UNITS, help='The number of hidden units')
    parser.add_argument('--rbm_k_gibbs_steps', type=int, default=RBM_K_GIBBS_STEPS, help='The number of steps in the Gibbs sampling')

    return parser


def run_pca(dataset_df: pd.DataFrame):
    pca = sklearn.decomposition.PCA(n_components=dataset_df.shape[1])
    dataset_pca_df = pca.fit_transform(dataset_df)

    return dataset_pca_df, pca


def unstandardize_results(results: pd.DataFrame, data_set: QoEDataset, n_columns: int) -> pd.DataFrame:
    """
    This function unnormalizes the labels by performing X * STD(X) + MEAN(X) performed in the process of dataset creation, thus it requires
    the original QoEDataset object
    :param results: pandas.DataFrame object containing the results with  2 * n_columns columns, where the first n_columns are the true, and the last n_columns are the predicted labels
    :param data_set: The original QoEDataset object which was created for the process of training / testing, and contains the mean and the std of each label
    :param n_columns: The number of labels
    :return: pandas.DataFrame containing the unnormalized true and predicted labels
    """
    print('Unstandardizing results...')
    unstandardized_results = pd.DataFrame()
    for line_idx in tqdm(range(len(results))):
        # - Get the line
        res = pd.DataFrame(results.iloc[line_idx]).T.reset_index(drop=True)

        # - Get the true labels
        labels = pd.DataFrame(data_set.unnormalize_labels(res.iloc[0, :n_columns].values)).T

        # - Rename the columns to include the "true" postfix
        for old_name in data_set.data_df.columns:
            new_name = f'{old_name}_true'
            labels = labels.rename(columns={old_name: new_name})

        # - Get the predictions
        preds = pd.DataFrame(data_set.unnormalize_labels(res.iloc[0, n_columns:].values)).T
        for old_name in data_set.data_df.columns:
            new_name = f'{old_name}_pred'
            preds = preds.rename(columns={old_name: new_name})

        # - Concatenate the labels with the preds horizontally
        labels_preds = pd.concat([labels, preds], axis=1)

        # - Append to the unnormalized_results
        unstandardized_results = pd.concat([unstandardized_results, labels_preds])

    # - Reset the index to normal
    unstandardized_results = unstandardized_results.reset_index(drop=True)

    return unstandardized_results


def get_number_of_parameters(model: torch.nn.Module):

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_non_trainable_parameters = n_total_params - n_trainable_params

    return n_trainable_params, n_non_trainable_parameters


def get_errors(results: pd.DataFrame, columns: list):
    n_columns = len(columns)
    true = results.iloc[:, :n_columns].values
    pred = results.iloc[:, n_columns:].values

    columns = [column_name + '_errors(%)' for column_name in columns]

    errors = pd.DataFrame(np.abs(100 - (true + EPSILON) * 100 / (pred + EPSILON)), columns=columns)

    return errors
