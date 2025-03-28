import os
import pathlib

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier

import numpy as np
import warnings

from zoom.data_division import DATA_ROOT_DIR

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import time
import sys
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))
sys.path.append(d)
from util.feature_extraction import FeatureExtractor
import matplotlib.pyplot as plt


OUTPUT_ROOT_DIR = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/outputs')
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)


class IP_UDP_ML:

    def __init__(self, feature_subset, estimator, config, metric, dataset, my_ip_l):
        self.feature_subset = feature_subset
        self.estimator = estimator
        self.config = config
        self.metric = metric
        self.feature_importances = {}
        self.feature_matrix = None
        self.target_vals = None
        self.dataset = dataset
        self.net_columns = ['frame.time_relative', 'frame.time_epoch', 'ip.src', 'ip.dst', 'ip.proto', 'ip.len',
                            'udp.srcport', 'udp.dstport', 'udp.length', 'rtp.ssrc', 'rtp.timestamp', 'rtp.seq', 'rtp.p_type', 'rtp.marker']
        self.my_ip_l = my_ip_l

    def train(self, list_of_files):

        feature_extractor = FeatureExtractor(self.feature_subset, self.config)
        print(
            f'\nExtracting features on training set...\nModel: {self.estimator.__class__.__name__}\nFeature Subset: {" ".join(self.feature_subset)}\nMetric: {self.metric}\n')

        t1 = time.time()

        train_data = []
        idx = 1
        total = len(list_of_files)
        for file_tuple in list_of_files:
            csv_file = file_tuple[0]
            labels_file = file_tuple[1]
            print(f'Extracting features for file # {idx} of {total}')
            df_net = pd.read_csv(csv_file)
            if df_net['ip.proto'].dtype == object:
                df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
            df_net = df_net[~df_net['ip.proto'].isna()]
            df_net['ip.proto'] = df_net['ip.proto'].astype(int)

            # calculates the ip_addr with the highest sum of 'udp.length' for each unique source IP
            ip_addr = df_net.groupby('ip.src').agg({'udp.length': sum}).reset_index(
            ).sort_values(by='udp.length', ascending=False).head(1)['ip.src'].iloc[0]
            if ip_addr in self.my_ip_l:  # if ip_addr is in my ip list, choose the ip with the 2nd highest udp length sum
                ip_addr = df_net.groupby('ip.src').agg({'udp.length': sum}).reset_index(
                ).sort_values(by='udp.length', ascending=False).head(2)['ip.src'].iloc[1]

            print('src ip:', ip_addr, ' - file name:', csv_file)
            # filters the DataFrame to retain only rows: 'ip.proto' is equal to 17, 'ip.src' is equal to ip_addr
            df_net = df_net[(df_net['ip.proto'] == 17) & (df_net['ip.src'] == ip_addr)]

            df_net = df_net[df_net['udp.length'] > 306]
            df_net = df_net.rename(columns={
                                   'udp.length': 'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
            df_net = df_net.sort_values(by=['time_normed'])
            df_net = df_net[['length', 'time', 'time_normed']]
            df_netml = feature_extractor.extract_features(df_net=df_net)

            df_labels = pd.read_csv(labels_file)
            if df_labels is None or len(df_net) == 0:
                idx += 1
                continue


            df_merged = pd.merge(df_netml, df_labels, on='et')
            df_merged = df_merged.drop(df_merged.index[0])  # Drop the first row
            df_merged = df_merged.drop(df_merged.index[-1:])  # Drop the last  row

            # filter samples
            if self.metric == 'fps':
                if 'screenshots_num' in df_merged.columns:  # Discarding samples with less than 45 screenshots
                    initial_count = len(df_merged)
                    df_merged = df_merged[df_merged['screenshots_num'] >= 30]
                    removed_count = initial_count - len(df_merged)
                    print(f'File: Removed {removed_count} samples with screenshots_num < 30')
                    df_merged = df_merged[df_merged.columns.difference(['screenshots_num'])]

            elif self.metric == 'quality-brisque':
                df_merged['quality-brisque'] = np.select([(df_merged['brisque'] < 20),
                                                          (df_merged['brisque'] >= 20) & (df_merged['brisque'] < 40),
                                                          (df_merged['brisque'] >= 40) & (df_merged['brisque'] < 60),
                                                          (df_merged['brisque'] >= 60) & (df_merged['brisque'] < 80)],
                                                         [1, 2, 3, 4], default=5)
                df_merged = df_merged[df_merged.columns.difference(['brisque'])]

            elif self.metric == 'quality-piqe':
                df_merged['quality-piqe'] = np.select([(df_merged['piqe'] < 21),
                                                       (df_merged['piqe'] >= 21) & (df_merged['piqe'] < 36),
                                                       (df_merged['piqe'] >= 36) & (df_merged['piqe'] < 51),
                                                       (df_merged['piqe'] >= 51) & (df_merged['piqe'] < 81)],
                                                      [1, 2, 3, 4], default=5)
                df_merged = df_merged[df_merged.columns.difference(['brisque', 'piqe'])]

            #if self.metric == 'brisque':
            #    df_merged['brisque'] = df_merged['brisque'].round().astype(int)

            fname = os.path.basename(file_tuple[0])
            #df_merged.to_csv(f'output_df_{fname[:-4]}_{self.metric}.csv', index=False)
            train_data.append(df_merged)
            idx += 1

        dur = round(time.time() - t1, 2)
        print(f'\nFeature extraction took {dur} seconds.\n')
        print('\nFitting the model...\n')
        X = pd.concat(train_data, axis=0)
        X = X.dropna()
        #X.to_csv(f'outputTrain.csv', index=False)
        print(X.shape)
        y = X[self.metric]
        X = X[X.columns.difference([self.metric, 'et', 'ts', 'file', 't_et', 'screenshots_num'])]
        self.feature_matrix = X.copy()
        self.target_vals = y.copy()
        if self.metric == 'fps' or self.metric == 'brisque':
            y = y.apply(lambda x: round(x))
        t1 = time.time()
        self.estimator.fit(X, y)
        dur = round(time.time() - t1, 2)
        print(f'\nModel training took {dur} seconds.\n')

        if isinstance(self.estimator, RandomForestRegressor) or isinstance(self.estimator, RandomForestClassifier)\
                or isinstance(self.estimator, XGBClassifier) or isinstance(self.estimator, XGBRegressor)\
                or isinstance(self.estimator, CatBoostClassifier) or isinstance(self.estimator, CatBoostRegressor):
            print('\nCalculating feature importance...\n')
            for idx, col in enumerate(X.columns):
                self.feature_importances[col] = self.estimator.feature_importances_[idx]

        output_dir = OUTPUT_ROOT_DIR / self.metric
        X.to_csv(output_dir / 'X.csv')
        y.to_csv(output_dir / 'y.csv')
        return X, y

    def estimate(self, file_tuple):
        csv_file = file_tuple[0]
        labels_file = file_tuple[1]
        feature_extractor = FeatureExtractor(
            feature_subset=self.feature_subset, config=self.config)
        print(csv_file)
        df_net = pd.read_csv(csv_file)
        if df_net['ip.proto'].dtype == object:
            df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
        df_net = df_net[~df_net['ip.proto'].isna()]
        df_net['ip.proto'] = df_net['ip.proto'].astype(int)

        # calculates the ip_addr with the highest sum of 'udp.length' for each unique source IP
        ip_addr = df_net.groupby('ip.src').agg({'udp.length': sum}).reset_index(
        ).sort_values(by='udp.length', ascending=False).head(1)['ip.src'].iloc[0]
        if ip_addr in self.my_ip_l:  # if ip_addr is in my ip list, choose the ip with the 2nd highest udp length sum
            ip_addr = df_net.groupby('ip.src').agg({'udp.length': sum}).reset_index(
            ).sort_values(by='udp.length', ascending=False).head(2)['ip.src'].iloc[1]

        print('src ip:', ip_addr)
        # filters the DataFrame to retain only rows: 'ip.proto' is equal to 17, 'ip.src' is equal to ip_addr
        df_net = df_net[(df_net['ip.proto'] == 17) & (df_net['ip.src'] == ip_addr)]

        df_net = df_net[df_net['udp.length'] > 306]
        df_net = df_net.rename(columns={
                               'udp.length': 'length', 'frame.time_epoch': 'time', 'frame.time_relative': 'time_normed'})
        df_net = df_net.sort_values(by=['time_normed'])
        df_net = df_net[['length', 'time', 'time_normed']]
        df_netml = feature_extractor.extract_features(df_net=df_net)

        # Shift the rows to get the features of the previous second
        #prev_second_features = df_netml.drop(columns=['et']).shift(1)
        # Shift the rows to get the features of the next second
        #next_second_features = df_netml.drop(columns=['et']).shift(-1)
        # Concatenate the original DataFrame with the shifted ones
        #df_netml = pd.concat([prev_second_features, df_netml], axis=1)
        #df_netml = pd.concat([prev_second_features, df_netml, next_second_features], axis=1)

        df_labels = pd.read_csv(labels_file)

        df_merged = pd.merge(df_netml, df_labels, on='et')
        df_merged = df_merged.drop(df_merged.index[0])  # Drop the first row
        df_merged = df_merged.drop(df_merged.index[-1:])  # Drop the last row

        if self.metric == 'fps':
            if 'screenshots_num' in df_merged.columns:
                df_merged = df_merged[df_merged.columns.difference(['screenshots_num'])]

        elif self.metric == 'quality-brisque':
            df_merged['quality-brisque'] = np.select([(df_merged['brisque'] < 20),
                                              (df_merged['brisque'] >= 20) & (df_merged['brisque'] < 40),
                                              (df_merged['brisque'] >= 40) & (df_merged['brisque'] < 60),
                                              (df_merged['brisque'] >= 60) & (df_merged['brisque'] < 80)],
                                              [1, 2, 3, 4], default=5)
            df_merged = df_merged[df_merged.columns.difference(['brisque'])]

        elif self.metric == 'quality-piqe':
            df_merged['quality-piqe'] = np.select([(df_merged['piqe'] < 21),
                                                      (df_merged['piqe'] >= 21) & (df_merged['piqe'] < 36),
                                                      (df_merged['piqe'] >= 36) & (df_merged['piqe'] < 51),
                                                      (df_merged['piqe'] >= 51) & (df_merged['piqe'] < 81)],
                                                     [1, 2, 3, 4], default=5)
            df_merged = df_merged[df_merged.columns.difference(['brisque', 'piqe'])]

        elif self.metric == 'brisque':
            df_merged['brisque'] = df_merged['brisque'].round().astype(int)

        X = df_merged
        X = X.dropna()
        timestamps = X['et']
        y_test = X[self.metric]
        X = X[X.columns.difference([self.metric, 'et', 'ts', 'file', 't_et'])]
        if X.shape[0] == 0:
            return None
        y_pred = self.estimator.predict(X)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = np.squeeze(y_pred)
        if self.metric == 'fps':
            y_pred = list(map(lambda x: round(x), y_pred))
            y_test = y_test.apply(lambda x: round(x))
        X[self.metric+'_ip-udp-ml'] = y_pred
        X[self.metric+'_gt'] = y_test
        X['timestamp'] = timestamps
        X['file'] = csv_file
        X['dataset'] = self.dataset
        return X[[self.metric+'_ip-udp-ml', self.metric+'_gt', 'timestamp', 'file', 'dataset']]

    def display_top5_features(self, name):
        # Sort the dictionary by values in descending order
        sorted_features = sorted(self.feature_importances.items(), key=lambda item: item[1], reverse=True)

        # Extract the top 5 features
        top_5_features = sorted_features[:5]

        # Separate the features and their importance values for plotting
        features, importances = zip(*top_5_features)

        # Plot the top 5 features by their importance
        plt.figure(figsize=(10, 6))
        plt.bar(features, importances, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Top 5 Feature Importances')
        #plt.show()
        plt.savefig(f'C:\\final_project\\notes and docs\\features importance\\fps by margin err - {name}.png')
        plt.close()
