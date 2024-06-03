import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
plt.style.use('ggplot')


DATA_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/QoE/data/zoom/encrypted_traffic')
data = pd.read_csv(DATA_ROOT / 'data.csv')
data = data.rename(columns={"Latancy": 'Latency'})
data.to_csv(DATA_ROOT / 'data_no_nan.csv')
data.head()
data = data.loc[~data.isna().loc[:, 'Bandwidth']]
data.head()
data_train = data.drop(columns=['NIQE'])
data_train.head()
data_trgt = data.loc[:, 'NIQE']
data_trgt

# - Get train / test split
TEST_PROP = 0.1
n_data = len(data)
n_test = int(n_data * TEST_PROP)
n_test
data_idxs = np.arange(n_data)
test_idxs = np.random.choice(data_idxs, n_test, replace=True)
test_idxs
test_data = data.iloc[test_idxs].reset_index(drop=True)
test_data
test_data.to_csv(DATA_ROOT / 'test_data.csv')
test_data = pd.read_csv(DATA_ROOT / 'test_data.csv')
test_data

train_data = data.iloc[np.setdiff1d(data_idxs, test_idxs)].reset_index(drop=True)
train_data
train_data.to_csv(DATA_ROOT / 'train_data.csv')
train_data = pd.read_csv(DATA_ROOT / 'train_data.csv')
train_data
# EDA

# - Bandwidth
bw = data.loc[:, 'Bandwidth']
x = np.arange(len(data))
plt.plot(x, bw)
plt.xlabel('Sample')
plt.ylabel('Bandwidth')

# - NIQE - target
niqe = data.loc[:, 'NIQE']
plt.plot(x, niqe)
plt.xlabel('Sample')
plt.ylabel('NIQE')

# - Resolution
res = data.loc[:, 'Resolution']
plt.plot(x, res)
plt.xlabel('Sample')
plt.ylabel('Resolution')

# - FPS
fps = data.loc[:, 'fps']
plt.plot(x, fps)
plt.xlabel('Sample')
plt.ylabel('FPS')


# - Latency
latancy = data.loc[:, 'Latancy']
plt.plot(x, latancy)
plt.xlabel('Sample')
plt.ylabel('Latency')


# - Jitter
jitter = data.loc[:, 'Jitter']
plt.plot(x, jitter)
plt.xlabel('Sample')
plt.ylabel('Jitter')

# - AVG time between packets
avg_t = data.loc[:, 'avg time between packets']
plt.plot(x, avg_t)
plt.xlabel('Sample')
plt.ylabel('Avg. time between packets')

# - Packets lenght
pckt_len = data.loc[:, 'packets length']
plt.plot(x, pckt_len)
plt.xlabel('Sample')
plt.ylabel('Packet length')

# - Cleaning cv_5_folds
data_norm = (data - data.mean()) / data.std()

data_norm.head()
data_norm.isna()

data_norm_no_na = data_norm.loc[~data_norm.isna().loc[:, 'Bandwidth']]

# - Corralation
data_corr = data_norm_no_na.corr()
data_corr
sns.heatmap(data_corr)
data_corr
# - PCA
data_no_niqe = data.drop(columns=['NIQE', 'Resolution', 'Dest_Port', 'Src_Port', 'Interval start'])
data_no_niqe.head()
data_no_niqe_norm = (data_no_niqe - data_no_niqe.mean()) / data.std()
data_no_niqe_norm = (data_no_niqe - data_no_niqe.mean()) / data_no_niqe.std()
data_no_niqe_norm.columns
pca = PCA(n_components=data_no_niqe_norm.shape[1])
pca.fit(data_no_niqe_norm)
exp_var = pca.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)
cum_exp_var

loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(len(data_no_niqe_norm.columns))], index=data_no_niqe_norm.columns)

loadings
loadings.sum(axis=0)
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Components')












