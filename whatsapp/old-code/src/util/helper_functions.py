from util.config import project_config
import math
import pandas as pd
import sys
from os.path import dirname, abspath
import numpy as np
import os
from sklearn.model_selection import KFold   # KfoldCVOverFiles
from collections import defaultdict
pd.set_option('display.float_format', lambda x: '%.2f' % x)
d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)


class KfoldCVOverFiles:

    def __init__(self, k, main_folder, net_conditions, metrics, random_state=None, shuffle=False):
        self.k = k
        self.main_folder = main_folder
        self.net_conditions = net_conditions
        self.metrics = metrics
        self.random_state = random_state
        self.shuffle = shuffle

    def split(self):
        # return a list of dict1 elements {net_cond_str: dict2}, dict2 is {metric: dict3}, dict3 is {"train"/"test": list of file tuples}

        cross_validation_splits = [{} for _ in range(self.k)]

        kf = KFold(n_splits=self.k, random_state=self.random_state, shuffle=self.shuffle)
        for net_cond in self.net_conditions:
            cond_folder = os.path.join(self.main_folder, net_cond)
            for metric in self.metrics:
                if metric == 'quality-brisque':
                    metric = 'brisque'
                # elif metric == 'piqe' or metric == 'quality-piqe':
                #     metric = 'brisque_piqe'

                print(f'net condition: {net_cond}\n', f'metric: {metric}\n----------------------')
                file_tuples_list = create_file_tuples_list_rtp(cond_folder, metric)
                X = np.array(file_tuples_list)
                idx = 1
                for train_index, test_index in kf.split(X):
                    if not net_cond in cross_validation_splits[idx - 1].keys():
                        cross_validation_splits[idx-1][net_cond] = {}
                    cross_validation_splits[idx - 1][net_cond][metric] = {}
                    X_train, X_test = list(X[train_index]), list(X[test_index])
                    cross_validation_splits[idx-1][net_cond][metric]['train'] = X_train
                    cross_validation_splits[idx-1][net_cond][metric]['test'] = X_test
                    print(
                        f'\nSplit # {idx} | net_cond = {net_cond} | metric = {metric} | n_files_train = {len(X_train)} | n_files_test = {len(X_test)} | train index = {train_index} | test index = {test_index}\n')
                    idx += 1
        return cross_validation_splits


def create_file_tuples_list(main_folder, metric):
    tuples_list = []
    if metric == 'quality-brisque':
        metric = 'brisque'
    elif metric == 'piqe' or metric == 'quality-piqe':
        metric = 'brisque_piqe'

    # Iterate over all folders in the main folder
    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            # Find the pcap file (starting with 'pcap' and ending with '.csv')
            pcap_file = next((f for f in os.listdir(folder_path) if f.startswith('pcap') and f.endswith('.csv')), None)

            # Find the labels file (starting with 'metric' and ending with '.csv')
            labels_file = next((f for f in os.listdir(folder_path) if f.startswith(metric) and f.endswith('.csv')),
                               None)

            # Add the tuple of paths to the list
            if pcap_file and labels_file:
                tuples_list.append((os.path.join(folder_path, pcap_file), os.path.join(folder_path, labels_file)))

    return tuples_list

def create_file_tuples_list_rtp(main_folder, metric):
    tuples_list = []
    if metric == 'quality-brisque':
        metric = 'brisque'
    # elif metric == 'piqe' or metric == 'quality-piqe':
    #     metric = 'brisque_piqe'

    # Iterate over all folders in the main folder
    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            # Find the pcap file (starting with 'pcap' and ending with '.csv')
            pcap_file = next((f for f in os.listdir(folder_path) if f.startswith('pcap') and f.endswith('Rtp.csv')), None)

            # Find the labels file (starting with 'metric' and ending with '.csv')
            labels_file = next((f for f in os.listdir(folder_path) if f.startswith(metric) and f.endswith('.csv')),
                               None)

            # Add the tuple of paths to the list
            if pcap_file and labels_file:
                tuples_list.append((os.path.join(folder_path, pcap_file), os.path.join(folder_path, labels_file)))

    return tuples_list


def filter_ptype(x):
    if type(x) != str and math.isnan(x):
        return x
    x = str(x)
    if ',' in x:
        return str(int(float(x.split(',')[0])))
    return str(int(float(x)))

'''
def mark_video_frames(pcap):
    pcap["is_video_pred"] = (
        pcap["udp.length"] > project_config['video_thresh']).astype(np.int32)
    return pcap


def filter_video_frames(pcap):
    pcap = pcap[pcap["udp.length"] > project_config['video_thresh']]
    return pcap
'''

'''
def filter_video_frames_rtp(pcap, vca):
    # if vca == 'webex':
    #     top_num = 1
    # else: 
    #     top_num = 2
    top_num = 1
    pcap['rtp.p_type'] = pcap['rtp.p_type'].apply(filter_ptype)
    top_x = pcap.groupby(['rtp.p_type'])['udp.length'].mean().nlargest(top_num).index.tolist()
    condition = ((pcap['rtp.p_type'].isin(top_x)))
    return pcap[condition]
'''

'''
def read_net_file(dataset, filename):
    csv_columns = ['frame.time_relative', 'frame.time_epoch', 'ip.src', 'ip.dst', 'ip.proto', 'ip.len', 'udp.srcport', 'udp.dstport', 'udp.length', 'rtp.ssrc', 'rtp.timestamp', 'rtp.seq', 'rtp.p_type', 'rtp.marker']
    df_net = pd.read_csv(filename)
    try:
        ip_addr = df_net.groupby('ip.dst').agg({'udp.length': sum}).reset_index().sort_values(by='udp.length', ascending=False).head(1)['ip.dst'].iloc[0]
    except IndexError:
        return
    df_net = df_net[(df_net["ip.dst"] == ip_addr) & (~pd.isna(df_net["rtp.ssrc"]))]
    df_net = df_net[~df_net['ip.proto'].isna()]
    df_net['rtp.p_type'] = df_net['rtp.p_type'].apply(filter_ptype)
    df_net['rtp.p_type'] = df_net['rtp.p_type'].dropna()
    df_net['ip.proto'] = df_net['ip.proto'].astype(str)
    df_net = df_net[df_net['ip.proto'].str.contains(',') == False]
    df_net['ip.proto'] = df_net['ip.proto'].apply(lambda x: int(float(x)))

    if df_net.empty:
        return
    return df_net
'''

'''
def get_net_stats(df_video, ft_end_col="frame_et"):
    ## frame duration calculations
    df_video = df_video.sort_values(by=ft_end_col)
    df_video["frame_size"] = df_video["frame_size"].apply(lambda x: float(x))
    df_video["frame_dur"] = df_video[ft_end_col].diff()

    df_video["avg_frame_dur"] = df_video["frame_dur"].rolling(30).mean()
    df_video = df_video.fillna(0)
    
    df_video["frame_dur"] = df_video["frame_dur"].apply(lambda x: 0 if x < 0 else x)

    ## freeze calculation
    df_video["is_freeze"] = df_video.apply(is_freeze, axis=1)
    df_video["freeze_dur"] = df_video.apply(get_freeze_dur, axis=1)
    
    ## obtain per second stats
    df_video["frame_et_int"] = df_video[ft_end_col].apply(lambda x: int(x)+1)
    df_grp = df_video.groupby("frame_et_int").agg({"frame_size" : ["sum", "count"], "is_freeze": "sum", 
                                             "freeze_dur": "sum", "frame_dur": "std"}).reset_index()
    
    ## rename columns
    df_grp.columns = ['_'.join(col).strip('_') for col in df_grp.columns.values]    
    df_grp = df_grp.rename(columns={'frame_size_count': 'predicted_framesReceivedPerSecond',
                                    'is_freeze': 'freeze_count',
                                    'frame_size_sum': 'predicted_bitrate',
                                    'freeze_dur': 'freeze_dur',
                                    'lost_frame': 'frames_lost',
                                    'frame_dur_std': 'predicted_frame_jitter'
                                   })
    df_grp['predicted_bitrate'] = df_grp['predicted_bitrate']*8
    df_grp['predicted_frame_jitter'] = df_grp['predicted_frame_jitter']*1000
    return df_grp
'''