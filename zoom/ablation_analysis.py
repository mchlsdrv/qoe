import os
import pandas as pd
import numpy as np
import pathlib


# === 10 Epochs ===
ABLATION_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/parameter_selection')
RESULTS_FILE_010_EPCH = ABLATION_ROOT / 'outputs_010_epochs/ablation_final_results.csv'
RESULTS_FILE_010_EPCH.is_file()

res_fl = pd.read_csv(RESULTS_FILE_010_EPCH)
res_gb = res_fl.groupby('dataset').min()
res_fl.loc[:, 'NIQE_errors(%)'].values.min()
res_fl.iloc[res_fl.loc[:, 'NIQE_errors(%)'].values.argmin()]
res_fl.loc[:, 'Resolution_errors(%)'].values.min()
res_fl.loc[res_fl.loc[:, 'Resolution_errors(%)'].values.argmin()]
res_fl.loc[:, 'fps_errors(%)'].values.min()
res_fl.loc[res_fl.loc[:, 'fps_errors(%)'].values.argmin()]

# === 50 Epochs ===
ABLATION_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/parameter_selection')
RESULTS_FILE_050_EPCH = ABLATION_ROOT / 'outputs_050_epochs/ablation_final_results.csv'
RESULTS_FILE_050_EPCH.is_file()

res_fl = pd.read_csv(RESULTS_FILE_050_EPCH)
res_fl.head()
res_fl.loc[:, 'NIQE_errors(%)'].values.min()
res_fl.iloc[res_fl.loc[:, 'NIQE_errors(%)'].values.argmin()]
res_fl.loc[:, 'Resolution_errors(%)'].values.min()
res_fl.loc[res_fl.loc[:, 'Resolution_errors(%)'].values.argmin()]
res_fl.loc[:, 'fps_errors(%)'].values.min()
res_fl.loc[res_fl.loc[:, 'fps_errors(%)'].values.argmin()]

# === 100 Epochs ===
ABLATION_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/parameter_selection')
RESULTS_FILE_100_EPCH = ABLATION_ROOT / 'outputs_100_epochs/ablation_final_results.csv'
RESULTS_FILE_100_EPCH.is_file()

res_fl = pd.read_csv(RESULTS_FILE_100_EPCH)
res_fl.head()
res_fl.loc[:, 'NIQE_errors(%)'].values.min()
res_fl.iloc[res_fl.loc[:, 'NIQE_errors(%)'].values.argmin()]
res_fl.loc[:, 'Resolution_errors(%)'].values.min()
res_fl.loc[res_fl.loc[:, 'Resolution_errors(%)'].values.argmin()]
res_fl.loc[:, 'fps_errors(%)'].values.min()
res_fl.loc[res_fl.loc[:, 'fps_errors(%)'].values.argmin()]
