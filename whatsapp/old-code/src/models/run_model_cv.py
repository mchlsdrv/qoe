from collections import defaultdict
from itertools import product
from datetime import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.utils.multiclass import type_of_target
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
import gc

import os
import time
import sys
from os.path import dirname, abspath, basename
DATA_ROOT_DIR = '/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/data'
project_root = dirname(dirname(abspath(__file__)))
sys.path.append(project_root)
print(project_root)
print(sys.path)
from util.config import project_config
import pickle
from models.ip_udp_ml import IP_UDP_ML
from models.rtp_ml import RTP_ML
from util.helper_functions import create_file_tuples_list, KfoldCVOverFiles, create_file_tuples_list_rtp


class ModelRunner:

    def __init__(self, metric, estimation_method, feature_subset, data_dir, cv_index, my_ip_l,
                 net_conditions_train, net_conditions_test):

        self.metric = metric  # label
        self.estimation_method = estimation_method  # model name
        # RandomForestRegressor on brisque and piqe score metric
        self.estimator = RandomForestClassifier() if self.metric == 'fps' or self.metric == 'quality-brisque' \
                                                     or self.metric == 'quality-piqe' else RandomForestRegressor()
        #self.estimator = XGBClassifier() if self.metric == 'fps' or self.metric == 'quality-brisque' \
        #                                    or self.metric == 'quality-piqe' else XGBRegressor()
        #self.estimator = CatBoostClassifier(logging_level='Silent') if self.metric == 'resolution' or self.metric == 'fps'\
        #                                    or self.metric == 'quality' else CatBoostRegressor(logging_level='Silent')
        # features subset from ['SIZE' 'IAT', 'LSTATS', 'TSTATS']
        self.feature_subset = 'none' if feature_subset is None else feature_subset
        self.data_dir = data_dir

        if feature_subset:
            feature_subset_tag = '-'.join(feature_subset)
        else:
            feature_subset_tag = 'none'

        net_cond_train_tag = '-'.join(net_conditions_train)
        net_cond_test_tag = '-'.join(net_conditions_test)

        data_bname = os.path.basename(data_dir)

        self.trial_id = '_'.join(
            [metric, net_cond_train_tag, self.estimation_method,
             net_cond_test_tag, f'cv_idx{cv_index}'])

        self.intermediates_dir = f'{self.data_dir}_intermediates/{self.trial_id}'

        self.cv_index = cv_index

        self.model = None

        self.my_ip_l = my_ip_l

    def save_intermediate(self, data_object, pickle_filename):
        pickle_filename = f'{self.trial_id}_{pickle_filename}'
        pickle_filepath = f'{self.intermediates_dir}/{pickle_filename}.pkl'

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(pickle_filepath), exist_ok=True)

        # Save the intermediate data object to a pickle file
        with open(pickle_filepath, 'wb') as fd1:
            pickle.dump(data_object, fd1)

    def load_intermediate(self, pickle_filename):
        with open(f'{self.intermediates_dir}/{pickle_filename}.pkl', 'rb') as fd:
            data_object = pickle.load(fd)
        return data_object

    def fps_prediction_accuracy(self, pred, truth, margin_err):
        # check accuracy of frame per second prediction
        # correct if |pred - label| <= 2, else incorrect
        n = len(pred)
        df = pd.DataFrame({'pred': pred.to_numpy(), 'truth': truth.to_numpy()})
        df['deviation'] = df['pred'] - df['truth']
        df['deviation'] = df['deviation'].abs()
        return len(df[df['deviation'] <= margin_err]) / n

    def bps_prediction_accuracy(self, pred, truth):
        # check accuracy of bit per second prediction
        # correct if |pred - label| <= 10% of label, else incorrect
        n = len(pred)
        df = pd.DataFrame({'pred': pred.to_numpy(), 'truth': truth.to_numpy()})
        df['abs_diff'] = (df['pred'] - df['truth']).abs()
        df['threshold'] = df['truth'] * 0.1
        correct_predictions = df[df['abs_diff'] <= df['threshold']]
        # Return the accuracy
        return len(correct_predictions) / n

    def brisque_prediction_accuracy(self, pred, truth):
        # check accuracy of brisque prediction
        # correct if |pred - label| <= 5, else incorrect
        n = len(pred)
        df = pd.DataFrame({'pred': pred.to_numpy(), 'truth': truth.to_numpy()})
        df['deviation'] = df['pred'] - df['truth']
        df['deviation'] = df['deviation'].abs()
        return len(df[df['deviation'] <= 5]) / n

    def piqe_prediction_accuracy(self, pred, truth):
        # check accuracy of piqe prediction
        # correct if |pred - label| <= 5, else incorrect
        n = len(pred)
        df = pd.DataFrame({'pred': pred.to_numpy(), 'truth': truth.to_numpy()})
        df['deviation'] = df['pred'] - df['truth']
        df['deviation'] = df['deviation'].abs()
        return len(df[df['deviation'] <= 5]) / n

    def train_model(self, split_files):
        bname = os.path.basename(self.data_dir)

        if self.estimation_method == 'ip-udp-ml':

            model = IP_UDP_ML(
                feature_subset=self.feature_subset,
                estimator=self.estimator,
                config=project_config,
                metric=self.metric,
                dataset=bname,
                my_ip_l=self.my_ip_l
            )
            model.train(split_files)

        elif self.estimation_method == 'rtp-ml':

            model = RTP_ML(
                feature_subset=self.feature_subset,
                estimator=self.estimator,
                config=project_config,
                metric=self.metric,
                dataset=bname,
                my_ip_l=self.my_ip_l
            )
            model.train(split_files)

        vca_model = model
        #self.save_intermediate(vca_model, 'vca_model')
        return vca_model

    def get_test_set_predictions(self, split_files, vca_model, low_margin, high_margin):
        predictions = []
        maes = []
        accs = []
        r2_scores = []
        acc_margin_lists = []
        acc_by_margin = {}

        idx = 1
        total = len(split_files)
        for file_tuple in split_files:
            #print(file_tuple[0])
            model = vca_model
            print(
                f'Testing {self.estimation_method} on file {idx} out of {total}...')
            output = model.estimate(file_tuple)

            output = output.dropna()

            if output is None:
                idx += 1
                predictions.append(output)
                continue

            # if the model isn't classifier calculate MAE and R2 score
            if self.metric != 'quality-brisque' and self.metric != 'quality-piqe':
                mae = mean_absolute_error(
                    output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                print(f'MAE = {round(mae, 2)}')
                maes.append(mae)

                r2 = r2_score(
                    output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                print(f'R2 score = {round(r2, 2)}')
                r2_scores.append(r2)

            else:  # classifier model: calculate classification accuracy
                acc = accuracy_score(
                    output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                print(f'Accuracy = {round(acc, 2) * 100}%')
                accs.append(acc)

            # calculate fps prediction accuracy (correct: absolute difference <= 2)
            if self.metric == 'fps':
                acc = self.fps_prediction_accuracy(
                    output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'], 2)
                accs.append(acc)
                print(f'Accuracy = {round(acc, 2) * 100}%')

                acc_margin_lists.append(self.fps_acc_by_margin_error(output, low_margin, high_margin))

            # calculate brisque prediction accuracy (correct: absolute difference <= 5)
            if self.metric == 'brisque':
                acc = self.brisque_prediction_accuracy(
                    output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                accs.append(acc)
                print(f'Accuracy = {round(acc, 2) * 100}%')

            # calculate piqe prediction accuracy (correct: absolute difference <= 5)
            if self.metric == 'piqe':
                acc = self.piqe_prediction_accuracy(
                    output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
                accs.append(acc)
                print(f'Accuracy = {round(acc, 2) * 100}%')

            idx += 1
            predictions.append(output)
            print("---------\n")

        if self.metric == 'fps':
            for margin_err in range(low_margin, high_margin):
                acc_sum = 0
                for l in acc_margin_lists:
                    acc_sum += l[margin_err]
                acc_by_margin[margin_err] = round(100 * acc_sum / len(acc_margin_lists), 2)

        if self.metric == 'quality-brisque' or self.metric == 'quality-piqe':
            mae_avg = "None"
            r2_avg = "None"
        else:
            mae_avg = round(sum(maes) / len(maes), 2)
            r2_avg = round(sum(r2_scores) / len(r2_scores), 2)
        #accuracy_str = ''
        acc_avg = round(100 * sum(accs) / len(accs), 2)
        accuracy_str = f'|| Accuracy_avg = {acc_avg}'
        line = f'{dt.now()}\tExperiment : {self.trial_id} || MAE_avg = {mae_avg} || R2_avg = {r2_avg} {accuracy_str}\n'
        with open('C:\\final_project\git_repo\data_collection_intermediates\\final-log-rf.txt', 'a') as fd1:
            fd1.write(line)

        #self.save_intermediate(predictions, 'predictions')
        return predictions, mae_avg, r2_avg, acc_avg, acc_by_margin

    def get_avg_cv_predictions(self, output):

        maes = []
        accs = []
        r2_scores = []

        # if the model isn't classifier calculate MAE and R2 score
        if self.metric != 'quality-piqe' or self.metric != 'quality-brisque':
            mae = mean_absolute_error(
                output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
            print(f'MAE = {round(mae, 2)}')
            maes.append(mae)

            r2 = r2_score(
                output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
            print(f'R2 score = {round(r2, 2)}')
            r2_scores.append(r2)

        else:  # classifier model: calculate classification accuracy
            a = accuracy_score(
                output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
            print(f'Accuracy = {round(a, 2) * 100}%')
            accs.append(a)

        # calculate fps prediction accuracy (correct: absolute difference <= 2)
        if self.metric == 'fps':
            acc = self.fps_prediction_accuracy(
                output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'], 2)
            accs.append(acc)
            print(f'Accuracy = {round(acc, 2) * 100}%')

        # calculate brisque prediction accuracy (correct: absolute difference <= 5)
        if self.metric == 'brisque':
            acc = self.brisque_prediction_accuracy(
                output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
            accs.append(acc)
            print(f'Accuracy = {round(acc, 2) * 100}%')

        # calculate piqe prediction accuracy (correct: absolute difference <= 5)
        if self.metric == 'piqe':
            acc = self.piqe_prediction_accuracy(
                output[f'{self.metric}_gt'], output[f'{self.metric}_{self.estimation_method}'])
            accs.append(acc)
            print(f'Accuracy = {round(acc, 2) * 100}%')

        print("---------\n")

        if self.metric == 'quality-brisque' or self.metric == 'quality-piqe':
            mae_avg = "None"
            r2_avg = "None"
        else:
            mae_avg = round(sum(maes) / len(maes), 2)
            r2_avg = round(sum(r2_scores) / len(r2_scores), 2)
        accuracy_str = ''
        acc_avg = round(100 * sum(accs) / len(accs), 2)
        accuracy_str = f'|| Accuracy_avg = {acc_avg}'
        line = f'{dt.now()}\tExperiment : {self.trial_id[:-1]}avg || MAE_avg = {mae_avg} || R2_avg = {r2_avg} {accuracy_str}\n'
        with open('C:\\final_project\git_repo\data_collection_intermediates\\final-log-avg-rf.txt', 'a') as fd:
            fd.write(line)

        #self.save_intermediate(output, 'predictions_cv-avg')


    def fps_acc_by_margin_error(self, output, low_margin, high_margin):
        acc_list = []
        for margin_err in range(low_margin, high_margin):
            acc_list.append(self.fps_prediction_accuracy(output[f'{self.metric}_gt'],
                                                                 output[f'{self.metric}_{self.estimation_method}'],
                                                                 margin_err))
        return acc_list


    def hyperparameter_tuning(self, metric):
        # Hyperparameter tuning for random forest model

        file_tuples_list = []
        for cond_dir in os.listdir(self.data_dir):
            cond_dir_path = os.path.join(self.data_dir, cond_dir)
            # for dir in data_dir:
            if not cond_dir_path.endswith('.DS_Store'):
                file_tuples_list += create_file_tuples_list(cond_dir_path, metric)

        bname = os.path.basename(self.data_dir)

        if self.estimation_method == 'ip-udp-ml':

            model = IP_UDP_ML(
                feature_subset=self.feature_subset,
                estimator=self.estimator,
                config=project_config,
                metric=self.metric,
                dataset=bname,
                my_ip_l=self.my_ip_l
            )
        else:  # self.estimation_method == 'rtp-ml':

            model = RTP_ML(
                feature_subset=self.feature_subset,
                estimator=self.estimator,
                config=project_config,
                metric=self.metric,
                dataset=bname,
                my_ip_l=self.my_ip_l
            )

        train_features, train_labels = model.train(file_tuples_list)

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=5)]
        # Number of features to consider at every split
        max_features = ['sqrt', 'log2', None]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 100, num=5)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(random_grid)

        # Use the random grid to search for best hyperparameters
        # Choose cross-validation strategy based on target type
        if type_of_target(train_labels) in ['binary', 'multiclass']:
            skf = StratifiedKFold(n_splits=3)
        else:  # target_type is 'continuous' for regression
            skf = KFold(n_splits=3)

        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=self.estimator, param_distributions=random_grid, n_iter=2, cv=skf, verbose=2,
                                       random_state=42, n_jobs=-1)

        # Fit the random search model
        rf_random.fit(train_features, train_labels)
        print(rf_random.best_params_)
        best_random = rf_random.best_estimator_

        # Save the best model
        #joblib.dump(best_random, f"best_random_forest_{metric}.pkl")

        # Clear memory
        del train_features, train_labels, rf_random, file_tuples_list
        gc.collect()

        return best_random


def plot_acc_by_margin_err(data, name):
    # Extracting keys and values from the dictionary
    margin_errors = list(data.keys())
    accuracies = list(data.values())

    # Convert accuracy values to integers for yticks
    yticks = [int(value) for value in accuracies]

    # Creating the plot
    plt.figure(figsize=(10, 6))
    plt.plot(margin_errors, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Model Accuracy by Margin Error')
    plt.xlabel('Margin Error')
    plt.ylabel('Accuracy Percentage')
    plt.grid(True)
    plt.xticks(margin_errors)
    plt.yticks(range(min(yticks), max(yticks) + 1, 5))  # Adjusting the step to 5 for better readability

    plt.savefig(f'C:\\final_project\\notes and docs\\final fps by margin err in order\\fps by margin err - {name}.png')
    plt.close()


if __name__ == '__main__':

    my_ip_l = ['10.100.102.32', '192.168.0.102', '10.0.0.115', '192.168.0.100', '192.168.0.103', '192.168.0.104']
    metrics = ['fps', 'brisque', 'piqe']
    # metrics = ['fps', 'brisque', 'piqe', 'quality-piqe', 'quality-brisque']
    estimation_methods = ['ip-udp-ml']  # model selection: ['ip-udp-ml', rtp-ml]

    # groups of features as per `features.feature_extraction.py`
    feature_subsets = [['LSTATS', 'TSTATS']]
    # network conditions set
    net_conditions = ["loss", "falls", "bandwidth"]

    net_conditions_train_test = [
                                (["bandwidth", "falls", "loss"], ["bandwidth"]),
                                (["bandwidth", "falls", "loss"], ["falls"]),
                                (["bandwidth", "falls", "loss"], ["loss"])
                                ]

    data_dir = [DATA_ROOT_DIR]

    low_margin_err, high_margin_err = 0, 5

    bname = os.path.basename(data_dir[0])

    # Create a directory for saving model intermediates
    intermediates_dir = f'{data_dir[0]}_intermediates'
    Path(intermediates_dir).mkdir(exist_ok=True, parents=True)

    # Create 5-fold cross validation splits and validate files. Refer `util/validator.py` for more details

    kcv = KfoldCVOverFiles(5, data_dir[0], net_conditions, metrics, None, False)  # random: seed num, True
                                                                                  # by order: None, False
    file_splits = kcv.split()

    #with open(f'{intermediates_dir}/cv_splits.pkl', 'wb') as fd:
    #    pickle.dump(file_splits, fd)

    #vca_preds = defaultdict(list)

    param_list = [metrics, estimation_methods, feature_subsets, net_conditions_train_test, data_dir]
    # Run models over 5 cross validations
    n = 1
    for metric, estimation_method, feature_subset, net_conditions_train_test, data_dir in product(*param_list):

        net_conds_subset_train = net_conditions_train_test[0]
        net_conds_subset_test = net_conditions_train_test[1]
        line = f'\n===================================================\n' \
               f'Train net condition: {" ".join(net_conds_subset_train)}\n' \
               f'Test net condition: {" ".join(net_conds_subset_test)}\n' \
               f'Metric: {metric}\n' \
               f'Estimation_method: {estimation_method}\n'
        with open('C:\\final_project\git_repo\data_collection_intermediates\\final-log-avg-rf.txt', 'a') as fd:
            fd.write(line)
        with open('C:\\final_project\git_repo\data_collection_intermediates\\final-log-rf.txt', 'a') as fd:
            fd.write(line)
        print(line)

        # hyperparameter tuning:
        model_runner = ModelRunner(
            metric, estimation_method, feature_subset, data_dir, 1, my_ip_l, net_conds_subset_train,
            net_conds_subset_test)
        best_model = model_runner.hyperparameter_tuning(metric)

        vca_preds = []
        #metric_net_cond_preds = []
        models = []
        acc_list = []
        r2_list = []
        mae_list = []
        acc_by_margin_list = []
        cv_idx = 1
        for fsp in file_splits:

            if metric == 'quality-brisque':
                metric_s = 'brisque'
            elif metric == 'quality-piqe' or metric == 'piqe':
                metric_s = 'brisque_piqe'
            else:
                metric_s = metric
            # create train and test file tuples lists
            train_file_tuple_list = []
            for net_cond in net_conditions:
                if net_cond in net_conds_subset_train:
                    train_file_tuple_list += fsp[net_cond][metric_s]["train"]

            test_file_tuple_list = []
            for net_cond in net_conditions:
                if net_cond in net_conds_subset_test:
                    test_file_tuple_list += fsp[net_cond][metric_s]["test"]

            model_runner = ModelRunner(
                metric, estimation_method, feature_subset, data_dir, cv_idx, my_ip_l, net_conds_subset_train, net_conds_subset_test)

            # select hyperparameter tuning results estimator:
            model_runner.estimator = best_model

            vca_model = model_runner.train_model(train_file_tuple_list)
            vca_model.display_top5_features(" ".join(net_conds_subset_train) + " - " + " ".join(net_conds_subset_test))
            #Path(f'{intermediates_dir}/{model_runner.trial_id}').mkdir(exist_ok=True, parents=True)
            predictions, mae, r2, acc, acc_by_margin = model_runner.get_test_set_predictions(test_file_tuple_list, vca_model, low_margin_err, high_margin_err)
            mae_list.append(mae)
            r2_list.append(r2)
            acc_list.append(acc)
            acc_by_margin_list.append(acc_by_margin)
            #models.append(vca_model)
            #with open(f'{intermediates_dir}/{model_runner.trial_id}/model.pkl', 'wb') as fd:
            #    pickle.dump(vca_model, fd)
            preds = pd.concat(predictions, axis=0)
            vca_preds.append(preds)
            #with open(f'{intermediates_dir}/{model_runner.trial_id}/predictions.pkl', 'wb') as fd:
            #    pickle.dump(preds, fd)
            cv_idx += 1
            #metric_net_cond_preds.append(preds)

        # calculate results by average all cross validations
        if metric == 'fps':
            acc_by_margin_avg_dict = {}
            for margin_err in range(low_margin_err, high_margin_err):
                acc_sum = 0
                for d in acc_by_margin_list:
                    acc_sum += d[margin_err]
                acc_by_margin_avg_dict[margin_err] = round(acc_sum / (cv_idx - 1), 2)

            plot_acc_by_margin_err(acc_by_margin_avg_dict, " ".join(net_conds_subset_train) + " - " + " ".join(net_conds_subset_test))


        if metric != 'resolution' and metric != 'quality-brisque' and metric != 'quality-piqe':
            mae_avg = round(sum(mae_list)/(cv_idx-1), 2)
            r2_avg = round(sum([abs(i) for i in r2_list])/(cv_idx-1), 2)
        else:
            mae_avg = r2_avg = "None"
        acc_avg = round(sum(acc_list)/(cv_idx-1), 2)
        n += 1
        line = f'{dt.now()}\tExperiment : {metric}_{"-".join(net_conds_subset_train)} ' \
               f'|| MAE_avg = {mae_avg} || R2_avg = {r2_avg} || acc_avg = {acc_avg}\n'
        with open('C:\\final_project\git_repo\data_collection_intermediates\\final-log-avg-rf.txt', 'a') as fd:
            fd.write(line)

        print(f'===========================\n===========================\n'
              f'train net condition subset: {" ".join(net_conds_subset_train)}\n'
              f'test net condition subset: {" ".join(net_conds_subset_test)}\n'
              f'metric: {metric}\n'
              f'estimation_method: {estimation_method}\n'
              f'===========================\n===========================\n'
              )
        print(line)

        # calculate results from total predictions
        combine_preds = pd.concat(vca_preds, axis=0)
        combine_preds.to_csv(f'combine_preds-{metric}_{"-".join(net_conds_subset_train)}_{"-".join(net_conds_subset_test)}.csv', index=False)
        #model_runner.get_avg_cv_predictions(combine_preds)