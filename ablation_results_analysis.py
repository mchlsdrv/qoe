import os
import pathlib
import numpy as np
import pandas as pd


DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/data/zoom/encrypted_traffic/ablation/outputs_2024-04-15_14-51-35/ablation_final_results_tmp.csv')
DATA_FILE.is_file()

data_df = pd.read_csv(DATA_FILE)
data_df.head()
data_gb = data_df.groupby('initial_lr').apply('min')
data_gb

data_df.columns
lbl1 = 'NIQE_errors(%)'
lbl2 = 'Resolution_errors(%)'
lbl3 = 'fps_errors(%)'

min_idx = np.abs(data_df.loc[:, lbl1]).argmin()
best_niqe_conff = data_df.iloc[min_idx]
best_niqe_conff
min_idx = np.abs(data_df.loc[:, lbl2]).argmin()
best_niqe_conff = data_df.iloc[min_idx]
best_niqe_conff
min_idx = np.abs(data_df.loc[:, lbl3]).argmin()
best_niqe_conff = data_df.iloc[min_idx]
best_niqe_conff










