import os
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
plt.style.use('ggplot')
mpl.rc('font', weight='bold', size=14)


DATA_ROOT = pathlib.Path('./data')
# data_file_name = 'data_clean.csv'
data_file_name = 'data_no_nan.csv'
data = pd.read_csv(DATA_ROOT / data_file_name)
data = data.drop(columns=['Unnamed: 0'])
data.head()
data.describe()

# LABEL PLOTS
fig, ax = plt.subplots(1, 3, figsize=(40, 10))

# - NIQE
lbl_niqe = data.loc[:, 'NIQE'].values
niqe_y, niqe_x = np.histogram(lbl_niqe, density=False)
ax[0].bar(niqe_x[:-1], niqe_y, width=0.34)
ax[0].set(xlabel='NIQE', ylabel='Count')
ax[0].set_xticks(labels=np.round(niqe_x[:-1], 1), ticks=niqe_x[:-1], rotation=30.0)

# - R
lbl_r = data.loc[:, 'R']
r_y = lbl_r.value_counts()[::-1]
r_x = [
    '320x180',
    '480x270',
    '640x360',
    '800x450',
    '960x540',
    '1120x630',
    '1280x720',
]
ax[1].bar(r_bin_lbls, r_y)#, width=88.)
ax[1].set_xticks(labels=r_x, ticks=r_bin_lbls, rotation=30)
ax[1].set(xlabel='R', ylabel='Count')

# - FPS
lbl_fps = data.loc[:, 'FPS'].values
fps_y, fps_x = np.histogram(lbl_fps, density=False)
ax[2].bar(fps_x[:-1], fps_y, width=2.7)
ax[2].set(xlabel='FPS', ylabel='Count')
ax[2].set_xticks(labels=np.round(fps_x[:-1], 1), ticks=fps_x[:-1], rotation=30.0)
# fig.savefig('./outputs/label_dist.png')


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

# -- Packets length
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

# - Correlation
data_cbrt_corr = data_mean_cbrt.corr()
data_cbrt_corr

# - Correlation
data_corr = data_norm.corr()
data_corr
sns.heatmap(data_corr)
data_corr

# - PCA
data_norm_labels = data_norm.loc[:, ['NIQE', 'R', 'FPS']]

data_norm_labels.head()
sns.boxenplot(data_norm_labels)

data_norm_features = data_norm.loc[:, ['BW', 'PPS', 'ATP', 'PL', 'J', 'L', 'DP', 'SP', 'IS']]
data_norm_features = data_norm.loc[:, ['BW', 'PPS', 'ATP', 'PL', 'J', 'L' ]]

data_norm_features.head()
sns.boxenplot(data_norm_features)
stats.zscore(data_norm_features)

# OUTLIERS
sns.boxenplot(data_norm_features)
data_norm_features_no_outliers = data_norm_features.loc[(np.abs(stats.zscore(data_norm_features)) < 2).all(axis=1)]
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
# data_norm_features_no_outliers = data_norm_no_outliers.loc[:, ['BW', 'ATP', 'PL', 'PPS', 'J', 'L', 'DP', 'SP', 'IS', 'ATP']].reset_index(drop=True)
data_norm_features_no_outliers = data_norm_no_outliers.loc[:, ['BW', 'ATP', 'PL', 'PPS', 'J', 'L', 'DP', 'SP', 'IS']].reset_index(drop=True)
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

fig, ax = plt.subplots()

ax.plot(np.arange(0, len(cum_exp_var)), pca.explained_variance_ratio_)
ax.set(xlabel='Primary Component', ylabel='Cummulative Explained Variance')
ax.set_xticks(labels=np.arange(1, len(cum_exp_var) + 1), ticks=np.arange(0, len(cum_exp_var)))

sns.boxenplot(data_norm_features_no_outliers_pca)


data_norm_features_no_outliers_pca_labels = pd.concat([data_norm_features_no_outliers_pca, data_norm_labels_no_outliers], axis=1)
data_norm_features_no_outliers_pca_labels
sns.boxenplot(data_norm_features_no_outliers_pca_labels)
data_norm_features_no_outliers_pca_labels.to_csv('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/data_norm_no_outliers_pca.csv', index=False)
