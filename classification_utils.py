import os

import numpy as np
import pandas as pd
import pathlib
from sklearn.metrics import precision_recall_fscore_support

from qoe.data_division import build_test_datasets

DEBUG = False


def train_classifier(X, y, classifier):
    n_classes = len(np.unique(y.values))

    # - Create the model
    model = classifier(max_depth=n_classes)

    # - Fit the model on the train data
    model.fit(X.values, y.values.flatten())

    return model


def eval_classifier(X, y, model):
    # - Predict the test data
    y_pred = model.predict(X.values)
    precision, recall, f1, support = precision_recall_fscore_support(y.values.flatten(), y_pred)
    precision = precision.mean()
    recall = recall.mean()
    f1 = f1.mean()

    if DEBUG:
        print(f'''
    Scores:
        - Precision: {precision}
        - Recall: {recall}
        - F1 score: {f1}
        - Support: {support}
    ''')

    return precision, recall, f1, support


def to_categorical(data: pd.DataFrame or np.ndarray, label: str = None, classes: list = None, class_bins: list = None, class_thresholds: list = None, as_array: bool = True):
    """
    Converts categorical features represented by some number by a running index [0, 1, 2, 3, ... , n_classes]
    :param data: The pd.DataFrame to be divided
    :param label: The label of the feature to be divided, if None - the input is np.ndarray
    :param classes: (Optional) If provided - no new classes will be calculated
    :param class_bins: Specific bins designated by thresholds of each class
    :param class_thresholds: list of the thresholds for each category
    :param as_array: If to output the result as a np.ndarray
    :return: Transformed pd.DataFrame
    """
    # - Get only the label column
    x = data
    if isinstance(data, pd.DataFrame) and label is not None:
        x = data.loc[:, label].values
    x = x.flatten()

    # - Find the different classes in the data
    if classes is None:
        classes = np.unique(x)
    class_codes = np.arange(len(classes))
    if class_bins is not None:
        # - Assign the classes based on boundaries
        for cls, cls_bin in enumerate(class_bins):
            cls_th = np.mean(cls_bin)
            x[np.argwhere((cls_bin[0] < x) & (x <= cls_th))] = cls_bin[0]
            x[np.argwhere((cls_th > x) & (x <= cls_bin[1]))] = cls_bin[1]

        # - If the value is lower than the minimal class - assign it the minimal class
        x[np.argwhere(x < class_bins[0][0])] = class_bins[0][0]

        # - If the value is greater than the maximal class - assign it the maximal class
        x[np.argwhere(x > class_bins[-1][1])] = class_bins[-1][1]

    elif class_thresholds is not None:
        # - Get global range boundaries
        cls_min = classes[0]
        cls_max = classes[-1]

        # - Constructing ranges for different classes
        class_boundaries = []
        for idx, cls_th in enumerate(class_thresholds):
            if idx == 0:
                class_boundaries.append((cls_min, cls_th))
            else:
                class_boundaries.append((class_thresholds[idx - 1], cls_th))
        class_boundaries.append((class_thresholds[-1], cls_max))

        # - Assign the classes based on boundaries
        for cls_code, cls_bndr in zip(class_codes, class_boundaries):
            x[np.argwhere(x >= cls_bndr[0]) & np.argwhere(x < cls_bndr[1])] = cls_code
        # - Fix the class for cls_max, as it will not fit into the last range
        # as the maximal value is not included in the range
        x[np.argwhere(x == cls_max)] = class_boundaries[- 1][-1]

    else:
        # - Divide into all possible classes
        for cls_code, cls in zip(class_codes, classes):
            x[np.argwhere(x == cls)] = cls_code

    # - If to transfer it back to pd.DataFrame
    if not as_array:
        data.loc[:, label] = x
        x = data
    return x, classes, class_codes


def run_cv(data_df: pd.DataFrame, classifier, n_folds: int, features: list, label: str, class_thresholds: list or None, data_dir: pathlib.Path, output_dir: pathlib.Path):
    cv_root_dir = data_dir / f'cv_{n_folds}_folds'
    if not cv_root_dir.is_dir():
        os.makedirs(cv_root_dir, exist_ok=True)
        build_test_datasets(data_df, n_folds=n_folds, root_save_dir=cv_root_dir)

    results = pd.DataFrame(columns=['precision', 'recall', 'f1'], dtype=np.float32)
    cv_dirs = os.listdir(cv_root_dir)
    for cv_dir in cv_dirs:
        train_df = pd.read_csv(cv_root_dir / cv_dir / 'train_data.csv')
        X_train = train_df.loc[:, features]
        y_train = train_df.loc[:, [label]]
        y_train = to_categorical(data=y_train, label=label, class_thresholds=class_thresholds)
        model = train_classifier(X=X_train, y=y_train, classifier=classifier)

        test_df = pd.read_csv(cv_root_dir / cv_dir / 'test_data.csv')
        X_test = test_df.loc[:, features]
        y_test = test_df.loc[:, [label]]
        y_test = to_categorical(data=y_test, label=label, class_thresholds=class_thresholds)

        precision, recall, f1, _ = eval_classifier(X=X_test, y=y_test, model=model)
        results = pd.concat([results, pd.DataFrame(dict(precision=precision, recall=recall, f1=f1), index=pd.Index([0]))], ignore_index=True)

    results.to_csv(output_dir / 'final_results.csv')
    mean_precision = results.loc[:, "precision"].mean()
    mean_recall = results.loc[:, "recall"].mean()
    mean_f1 = results.loc[:, "f1"].mean()
    n_classes = len(np.unique(data_df.loc[:, label].values.flatten())) if class_thresholds is None else len(class_thresholds) + 1
    print(f'''
Mean Stats on {n_folds} CV for {n_classes} classes:
    Precision | Recall | F1 Score
    ----------|--------|---------
      {mean_precision:.3f}   |  {mean_recall:.3f} |  {mean_f1:.3f} 
    ''')
    return results
