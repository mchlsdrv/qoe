import os
import pathlib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

from regression_utils import normalize_columns

plt.style.use('ggplot')


def plot_feature_importance(
        data: list or np.ndarray,
        features: list,
        errors: list or np.ndarray,
        title: str,
        axes: plt.axes,
):
    print(f'{title}:')
    print('\t> Summary:')
    for feat, scr in zip(features, data):
        print(f'\t\t{feat} score = {scr:.3f}')

    if isinstance(errors, list) or isinstance(errors, np.ndarray):
        axes.bar(features, data, yerr=errors)
    else:
        axes.bar(features, data)

    # - If the value is large - display it on a logarithmic scale
    if np.mean(data) > 100:
        axes.set_yscale('log')

    # - Set labels
    axes.tick_params(axis='both', which='major', labelsize=90)
    axes.set_xlabel('Feature', fontsize=100)
    axes.set_ylabel('Importance', fontsize=100)
    axes.set_title(title, fontweight='bold', fontsize=120)

def calc_coefficient_importance(model, features, name: str, axes: plt.axes):
    try:
        coeffs = model.coef_.flatten()
    except Exception as err:
        print(err)
        coeffs = model.feature_importances_
    feats_coeffs = list(zip(features, coeffs))
    feats_coeffs = sorted(feats_coeffs, key=lambda x: np.abs(x[1]))[::-1]
    feats, coeffs = [x[0] for x in feats_coeffs], [x[1] for x in feats_coeffs]
    plot_feature_importance(
        data=coeffs,
        features=features,
        errors=None,
        title=f'{name}',
        axes=axes,
    )


def calc_permutation_importance(model, x, y, features, scoring: str, name: str, axes: plt.axes):
    permut_imp = permutation_importance(model, x, y, scoring=scoring)
    imp_mu, imp_std = permut_imp.get('importances_mean'), permut_imp.get('importances_std')

    feats_mus_stds = list(zip(features, imp_mu, imp_std))
    feats_mus_stds = sorted(feats_mus_stds, key=lambda val: val[1])[::-1]
    feats, imp_mu, imp_std = [x[0] for x in feats_mus_stds], [x[1] for x in feats_mus_stds], [x[2] for x in feats_mus_stds]
    plot_feature_importance(
        data=imp_mu,
        features=feats,
        errors=imp_std,
        title=f'{name}',
        axes=axes,
    )


def compute_feature_importance(model, x, y, features, name: str, scoring: str, axes: list):
    # - Coefficient importance
    calc_coefficient_importance(
        model=model,
        features=features,
        name=name,
        axes=axes[0]
    )

    # - Permutation test
    calc_permutation_importance(
        model=model,
        x=x,
        y=y,
        features=features,
        scoring=scoring,
        name=name,
        axes=axes[1],
    )




# LABEL = 'NIQE'
# LABEL = 'FPS'
LABEL = 'R'
SAVE_DIR = pathlib.Path(f'./outputs/feat_importance_{LABEL.lower()}')
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_DIR.is_dir()
REGRESSOR = LinearRegression
DATA_FILE = pathlib.Path('./data/data_no_nan.csv')
data_df = pd.read_csv(DATA_FILE)
data_df.head()
FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 3.14+/-2.815, 11.30+/-20.179, 10.89+/-23.477


data_df, _, _ = normalize_columns(data_df=data_df, columns=FEATURES)
X_TRAIN = data_df.loc[:, FEATURES].values
Y_TRAIN = data_df.loc[:, [LABEL]].values

coeff_fig, coeff_ax = plt.subplots(2, 4, figsize=(120, 50), constrained_layout=True)
permut_fig, permut_ax = plt.subplots(2, 4, figsize=(120, 50), constrained_layout=True)

# - Linear regression
# -- Coefficient test
mdl = LinearRegression()
mdl.fit(X_TRAIN, Y_TRAIN)
compute_feature_importance(
    model=mdl,
    x=X_TRAIN,
    y=Y_TRAIN,
    features=FEATURES,
    scoring='neg_mean_squared_error',
    name='Linear Regression',
    axes=[coeff_ax[0, 0], permut_ax[0, 0]]
)

# - Ridge regression
mdl = Ridge()
mdl.fit(X_TRAIN, Y_TRAIN)


compute_feature_importance(
    model=mdl,
    x=X_TRAIN,
    y=Y_TRAIN,
    features=FEATURES,
    scoring='neg_mean_squared_error',
    name='Ridge Regression',
    axes=[coeff_ax[0, 1], permut_ax[0, 1]]
)

# - Elastic Net Regression
mdl = ElasticNet()
mdl.fit(X_TRAIN, Y_TRAIN)

compute_feature_importance(
    model=mdl,
    x=X_TRAIN,
    y=Y_TRAIN,
    features=FEATURES,
    scoring='neg_mean_squared_error',
    name='Elastic Net Regression',
    axes=[coeff_ax[0, 2], permut_ax[0, 2]]
)

# - Decision Trees
mdl = DecisionTreeRegressor()
mdl.fit(X_TRAIN, Y_TRAIN)

compute_feature_importance(
    model=mdl,
    x=X_TRAIN,
    y=Y_TRAIN,
    features=FEATURES,
    scoring='neg_mean_squared_error',
    name='Decision Tree',
    axes=[coeff_ax[0, 3], permut_ax[0, 3]]
)

# - Random Forest
mdl = RandomForestRegressor()
mdl.fit(X_TRAIN, Y_TRAIN.flatten())
compute_feature_importance(
    model=mdl,
    x=X_TRAIN,
    y=Y_TRAIN,
    features=FEATURES,
    scoring='neg_mean_squared_error',
    name='Random Forest',
    axes=[coeff_ax[1, 0], permut_ax[1, 0]]
)

# - SVM
mdl = svm.SVR(kernel='linear')
mdl.fit(X_TRAIN, Y_TRAIN.flatten())
compute_feature_importance(
    model=mdl,
    x=X_TRAIN,
    y=Y_TRAIN,
    features=FEATURES,
    scoring='neg_mean_squared_error',
    name='SVM',
    axes=[coeff_ax[1, 1], permut_ax[1, 1]]
)

# - XGBoost
mdl = XGBRegressor()
mdl.fit(X_TRAIN, Y_TRAIN.flatten())
compute_feature_importance(
    model=mdl,
    x=X_TRAIN,
    y=Y_TRAIN,
    features=FEATURES,
    scoring='neg_mean_squared_error',
    name='XGBoost',
    axes=[coeff_ax[1, 2], permut_ax[1, 2]]
)


# - AdaBoost
mdl = AdaBoostRegressor()
mdl.fit(X_TRAIN, Y_TRAIN.flatten())
compute_feature_importance(
    model=mdl,
    x=X_TRAIN,
    y=Y_TRAIN,
    features=FEATURES,
    scoring='neg_mean_squared_error',
    name='AdaBoost',
    axes=[coeff_ax[1, 3], permut_ax[1, 3]]
)

coeff_fig.savefig(SAVE_DIR / f'{LABEL.lower()}_feature_importance_coeff_test.png')
permut_fig.savefig(SAVE_DIR / f'{LABEL.lower()}_feature_importance_permut_test.png')
