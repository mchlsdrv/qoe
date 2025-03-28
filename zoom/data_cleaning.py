import os
import pathlib
import numpy as np
import pandas as pd


DATA_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/')
data = pd.read_csv(DATA_ROOT / 'data.csv')
data.head()
data = data.rename(columns={
    'Bandwidth': 'BW',
    'Jitter': 'J',
    'Resolution': 'R',
    'Latancy': 'L',
    'avg time between packets': 'ATP',
    'fps': 'FPS',
    'pps': 'PPS',
    'packets length': 'PL',
    'Interval start': 'IS',
    'Src_Port': 'SP',
    'Dest_Port': 'DP',
})
data = data.loc[~data.isna().loc[:, 'BW']]
data.head()
data = data.astype({
    'BW': np.float32,
    'J': np.float32,
    'R': np.int16,
    'L': np.float32,
    'ATP': np.float32,
    'FPS': np.int16,
    'PPS': np.int16,
    'PL': np.int16,
    'IS': np.int16,
    'SP': np.int16,
    'DP': np.int16,
})
res_dict = {
    '1280': '1280x720',
    '1120': '1120x630',
    '960' : '960x540',
    '800' : '800x450',
    '640' : '640x360',
    '480' : '480x270',
    '320' : '320x180'
}
data.loc[:, 'R'] = data.loc[:, 'R'].apply(lambda x: res_dict.get(str(x)))
data.loc[:, 'R'] = data.loc[:, 'R'].astype(str)
data.loc[:, 'R'] = pd.Categorical(categories=)
data_save_file = DATA_ROOT / 'data_clean.csv'
data.to_csv(data_save_file, index=False)
data = pd.read_csv(data_save_file)
data.dtypes
data.head()
