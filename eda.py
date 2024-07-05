import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats
plt.style.use('ggplot')


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
data.to_csv(DATA_ROOT / 'data_no_nan.csv')
data.head()
data.describe()

# LABELS
# -- Resolution
lbl_r = data.loc[:, 'R']
sns.displot(lbl_r)
sns.boxenplot(lbl_r)

# -- FPS
lbl_fps= data.loc[:, 'FPS']
sns.histplot(lbl_fps)
sns.boxenplot(lbl_fps)
# -- NIQE
lbl_niqe = data.loc[:, 'NIQE']
sns.histplot(lbl_niqe)#, stat='density')#, binwidth=10)
sns.boxenplot(lbl_niqe)


# FEATURES
# - Bandwidth
feat_bw = data.loc[:, 'BW']
x = np.arange(len(data))
plt.plot(x, feat_bw)
sns.histplot(feat_bw)#, stat='density')#, binwidth=10)
sns.boxenplot(feat_bw)

# -- Latency
feat_latency = data.loc[:, 'L']
plt.plot(x, feat_latency)
sns.histplot(feat_latency)
sns.boxenplot(feat_latency)

# -- Jitter
feat_jitter = data.loc[:, 'J']
plt.plot(x, feat_jitter)
sns.histplot(feat_jitter)
sns.boxenplot(feat_jitter)
# -- AVG time between packets
feat_avg_tp = data.loc[:, 'ATP']
plt.plot(x, feat_avg_tp)
sns.histplot(feat_avg_tp)
sns.boxenplot(feat_avg_tp)
# -- Packets lenght
feat_pckt_len = data.loc[:, 'PL']
plt.plot(x, feat_pckt_len)
sns.histplot(feat_pckt_len)
sns.boxenplot(feat_pckt_len)

# - Data normalization
data.mean()
data_norm = (data - data.mean()) / data.std()
# - Check is there are any NAs in the data
data_norm.isna().sum()
data_norm.describe()
sns.boxenplot(data_norm)
data_norm.head()

# - Corralation
data_corr = data_norm.corr()
data_corr
sns.heatmap(data_corr)
data_corr

# - PCA
data_norm_labels = data_norm.loc[:, ['NIQE', 'R', 'FPS']]

data_norm_labels.head()
sns.boxenplot(data_norm_labels)

data_norm_features = data_norm.loc[:, ['BW', 'PPS', 'ATP', 'PL', 'J', 'L']]

data_norm_features.head()
sns.boxenplot(data_norm_features)
stats.zscore(data_norm_features)

# OUTLIERS
sns.boxenplot(data_norm_features)
data_norm_features_no_outliers = data_norm_features.loc[(np.abs(stats.zscore(data_norm_features)) < 3).all(axis=1)]
sns.boxenplot(data_norm_features_no_outliers)
L = len(data_norm_features)
N = len(data_norm_features_no_outliers)
R = 100 - N * 100 / L
print(f'''
Total before reduction: {L}
Total after reduction: {N}
> Reduced: {L - N} ({R:.3f}%)
    ''')


data_norm_no_outliers = data_norm.loc[(np.abs(stats.zscore(data_norm)) < 2).all(axis=1)]
sns.boxenplot(data_norm_no_outliers)
data_norm_features_no_outliers = data_norm_no_outliers.loc[:, ['BW', 'ATP', 'PL', 'PPS']].reset_index(drop=True)
data_norm_features_no_outliers
data_norm_labels_no_outliers = data_norm_no_outliers.loc[:, ['NIQE', 'R', 'FPS']].reset_index(drop=True)
data_norm_labels_no_outliers

pca = PCA(n_components=data_norm_features_no_outliers.shape[1])
data_norm_features_no_outliers_pca = pd.DataFrame(pca.fit_transform(data_norm_features_no_outliers))
data_norm_features_no_outliers_pca

loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i}' for i in range(len(data_norm_features_no_outliers.columns))], index=data_norm_features_no_outliers.columns)
loadings

exp_var = pca.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var)
cum_exp_var

plt.plot(1 - pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Components')

sns.boxenplot(data_norm_features_no_outliers_pca)


data_norm_features_no_outliers_pca_labels = pd.concat([data_norm_features_no_outliers_pca, data_norm_labels_no_outliers], axis=1)
data_norm_features_no_outliers_pca_labels
sns.boxenplot(data_norm_features_no_outliers_pca_labels)
data_norm_features_no_outliers_pca_labels.to_csv('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/data_norm_no_outliers_pca.csv', index=False)

