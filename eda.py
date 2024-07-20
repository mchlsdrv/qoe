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
data_file_name = 'data_clean.csv'
data = pd.read_csv(DATA_ROOT / data_file_name)
data.loc[:, 'R']
data.head()
data.describe()

# LABELS
# -- Resolution
lbl_r = data.loc[:, 'R']

res_order = [
    '320x180',
    '480x270',
    '640x360',
    '800x450',
    '960x540',
    '1120x630',
    '1280x720',
]
data.loc[:, 'R'] = pd.Categorical(data['R'], categories=res_order)
data.head()
res_order
g = sns.displot(lbl_r, col_order=res_order)
plt.xticks(res_order, rotation=30)

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

data_mean_log = np.log(data_norm)
sns.boxenplot(data_mean_log)
data_mean_sqrt = np.sqrt(data_norm)
sns.boxenplot(data_mean_sqrt)
data_mean_cbrt = np.cbrt(data_norm)
sns.boxenplot(data_mean_cbrt)
data_mean_cbrt.to_csv(DATA_ROOT / 'data_norm_cbrt.csv', index=False)
# - Corralation
data_cbrt_corr = data_mean_cbrt.corr()
data_cbrt_corr
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

