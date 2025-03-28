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
plt.style.use('ggplot')



def plot_feature_importance_one_plot(
        data: list or np.ndarray,
        features: list,
        errors: list or np.ndarray,
        title: str,
        figure: plt.figure,
        axes: plt.axes,
        save_file: pathlib.Path or str
):
    print('Summary:')
    for feat, scr in zip(features, data):
        print(f'\t{feat} score = {scr:.3f}')

    if isinstance(errors, list) or isinstance(errors, np.ndarray):
        axes.bar(features, data, yerr=errors)
    else:
        axes.bar(features, data)
    axes.set(title=title, xlab='Feature', ylab='Importance')

    return figure



def plot_feature_importance(data: list or np.ndarray, features: list, errors: list or np.ndarray, title: str, save_file: pathlib.Path or str):
    print('Summary:')
    for feat, scr in zip(features, data):
        print(f'\t{feat} score = {scr:.3f}')

    if isinstance(errors, list) or isinstance(errors, np.ndarray):
        plt.bar(features, data, yerr=errors)
    else:
        plt.bar(features, data)
    plt.xlabel('Feature')
    plt.title(title)
    plt.ylabel('Importance')
    plt.savefig(save_file)
    plt.show()
    plt.close()


def calc_coeff_importance(model, features, name: str, save_dir: pathlib.Path or str):
    try:
        coeffs = model.coef_.flatten()
    except Exception as err:
        print(err)
        coeffs = model.feature_importances_
    feats_coeffs = list(zip(features, coeffs))
    feats_coeffs = sorted(feats_coeffs, key=lambda x: x[1])[::-1]
    feats, coeffs = [x[0] for x in feats_coeffs], [x[1] for x in feats_coeffs]
    plot_feature_importance(
        data=coeffs,
        features=features,
        errors=None,
        title=f'{name} Coefficients',
        save_file=save_dir / f'{name}_coeffs.png'
    )


def calc_permutation_importance(model, x, y, features, scoring: str, name: str, save_dir: pathlib.Path or str):
    permut_imp = permutation_importance(model, x, y, scoring=scoring)
    imp_mu, imp_std = permut_imp.get('importances_mean'), permut_imp.get('importances_std')

    feats_mus_stds = list(zip(features, imp_mu, imp_std))
    feats_mus_stds = sorted(feats_mus_stds, key=lambda x: x[1])[::-1]
    feats, imp_mu, imp_std = [x[0] for x in feats_mus_stds], [x[1] for x in feats_mus_stds], [x[2] for x in feats_mus_stds]
    plot_feature_importance(
        data=imp_mu,
        features=feats,
        errors=imp_std,
        title=f'{name} Permutation Feature Importance',
        save_file=save_dir / f'{name}_permut.png'
    )


def compute_feature_importance(model, x, y, features, name: str, scoring: str, save_dir: pathlib.Path or str):
    calc_coeff_importance(
        model=model,
        features=features,
        name=name,
        save_dir=save_dir
    )
    # -- Permutation test
    calc_permutation_importance(
        model=model,
        x=x,
        y=y,
        features=features,
        scoring=scoring,
        name=name,
        save_dir=save_dir
    )



SAVE_DIR = pathlib.Path('outputs/feat_importance_niqe')
os.makedirs(SAVE_DIR)
REGRESSOR = LinearRegression
DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/Projects/PhD/QoE/Data/zoom/encrypted_traffic/data_no_nan.csv')
data_df = pd.read_csv(DATA_FILE)
data_df.head()
FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'L', 'J', 'DP', 'SP', 'IS']  # 3.14+/-2.815, 11.30+/-20.179, 10.89+/-23.477
LABEL = 'NIQE'


X_TRAIN = data_df.loc[:, FEATURES].values
Y_TRAIN = data_df.loc[:, [LABEL]].values


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
    save_dir=SAVE_DIR
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
    save_dir=SAVE_DIR
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
    save_dir=SAVE_DIR
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
    save_dir=SAVE_DIR
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
    save_dir=SAVE_DIR
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
    save_dir=SAVE_DIR
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
    save_dir=SAVE_DIR
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
    save_dir=SAVE_DIR
)


