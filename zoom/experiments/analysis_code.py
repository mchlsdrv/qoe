import datetime
import os
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from qoe.data_division import build_test_datasets


def to_categorical(data_df: pd.DataFrame, label: str, class_thresholds: list = None):
    """
    Converts categorical features represented by some number by a running index [0, 1, 2, 3, ... , n_classes]
    :param data_df: The pd.DataFrame to be divided
    :param label: The label of the feature to be divided
    :param class_thresholds: list of the thresholds for each category
    :return: Transformed pd.DataFrame
    """
    classes = np.unique(data_df.loc[:, label])
    if class_thresholds is not None:
        cls_min = classes[0]
        cls_max = classes[-1]

        class_boundaries = []
        for idx, cls_th in enumerate(class_thresholds):

            if idx == 0:
                class_boundaries.append((cls_min, cls_th))
            else:
                class_boundaries.append((class_thresholds[idx - 1], cls_th))
            # elif idx == len(class_thresholds) - 1:
        class_boundaries.append((class_thresholds[-1], cls_max))

        for cls, cls_bndr in enumerate(class_boundaries):
            data_df.loc[(data_df.loc[:, label] >= cls_bndr[0]) & (data_df.loc[:, label] < cls_bndr[1])] = cls
        # - Fix the class for cls_max
        data_df.loc[data_df.loc[:, label] == cls_max] = len(class_boundaries) - 1
    else:
        data_df.loc[:, label] = data_df.loc[:, label].apply(lambda x: np.argwhere(x == classes).flatten()[0])
    return data_df


DATA_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/PhD/QoE/data/zoom/encrypted_traffic/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path('/qoe/comparisons/classification/DCT')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
df = pd.read_csv(DATA_FILE)
df.head()
df.describe()

# - Split into features and labels
X = df.loc[:, ['BW', 'PPS', 'ATP', 'PL']]
# X = df.loc[:, ['BW', 'L', 'J', 'PPS', 'ATP', 'PL']]

y = df.loc[:, ['R']]
if N_CLASSES == 2:
    y.loc[:, 'R'] = y.loc[:, 'R'].apply(lambda x: 0 if x <= 800 else 1)
else:
    y.loc[:, 'R'] = y.loc[:, 'R'].apply(lambda x: np.argwhere(res == x).flatten()[0])

# - Transform resolution to category
res = np.unique(y.values.flatten())
y.loc[:, 'R'] = y.loc[:, 'R'].apply(lambda x: 0 if x <= 800 else 1)
# y.loc[:, 'R'] = y.loc[:, 'R'].apply(lambda x: np.argwhere(res == x).flatten()[0])
np.unique(y.values)

# - Split into train / train_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train
y_train
# - Create the model
dct = DecisionTreeClassifier(max_depth=len(res))

# - Fit the model on the train data
dct.fit(X_train.values, y_train.values)

# - Predict the train_test data
y_pred = dct.predict(X_test)

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
precision = precision.mean()
recall = recall.mean()
f1 = f1.mean()

print(f'''
Scores:
    - Precision: {precision}
    - Recall: {recall}
    - F1 score: {f1}
    - Support: {support}
''')
# - Evaluate the results
conf_mat = confusion_matrix(y_test, y_pred)

dct.score(X_test, y_test)
sns.heatmap(conf_mat)

print(classification_report(y_test, y_pred))
