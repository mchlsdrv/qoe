import os
import pathlib
import numpy as np
import pandas as pd


DATA_FILE = pathlib.Path('/home/mchlsdrv/Desktop/projects/qoe/outputs/outputs_2025-01-12_12-07-54/ablation_final_results.csv')
DATA_FILE.is_file()

data_df = pd.read_csv(DATA_FILE)
data_df.head()
err_vals = data_df.loc[:, 'NIQE_errors(%)'].values
err_vals.mean()
err_vals.std()
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
