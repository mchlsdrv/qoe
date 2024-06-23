import datetime
import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from qoe.classification_utils import run_cv


CLASSIFIER = RandomForestClassifier
DATA_FILE = pathlib.Path('/data/zoom/encrypted_traffic/data_no_nan.csv')
ROOT_OUTPUT_DIR = pathlib.Path('/qoe/comparisons/classification/RF')
TS = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FEATURES = ['BW', 'PPS', 'ATP', 'PL', 'J', 'L']
LABEL = 'R'
N_FOLDS = 10


if __name__ == '__main__':
    data_df = pd.read_csv(DATA_FILE)

    # - Binary
    class_thresholds = [800.]
    n_classes = len(class_thresholds) + 1
    output_dir = ROOT_OUTPUT_DIR / f'outputs_{N_FOLDS}_cv_{n_classes}_classes_{TS}'
    os.makedirs(output_dir, exist_ok=True)
    run_cv(
        data_df=data_df,
        classifier=CLASSIFIER,
        n_folds=N_FOLDS,
        features=FEATURES,
        label=LABEL,
        class_thresholds=class_thresholds,
        data_dir=DATA_FILE.parent,
        output_dir=output_dir
    )

    # - Trinary
    class_thresholds = [640., 960.]
    n_classes = len(class_thresholds) + 1
    output_dir = ROOT_OUTPUT_DIR / f'outputs_{N_FOLDS}_cv_{n_classes}_classes_{TS}'
    os.makedirs(output_dir, exist_ok=True)
    run_cv(
        data_df=data_df,
        classifier=CLASSIFIER,
        n_folds=N_FOLDS,
        features=FEATURES,
        label=LABEL,
        class_thresholds=class_thresholds,
        data_dir=DATA_FILE.parent,
        output_dir=output_dir
    )

    # - 4 classes
    class_thresholds = [480., 640., 960.]
    n_classes = len(class_thresholds) + 1
    output_dir = ROOT_OUTPUT_DIR / f'outputs_{N_FOLDS}_cv_{n_classes}_classes_{TS}'
    os.makedirs(output_dir, exist_ok=True)
    run_cv(
        data_df=data_df,
        classifier=CLASSIFIER,
        n_folds=N_FOLDS,
        features=FEATURES,
        label=LABEL,
        class_thresholds=class_thresholds,
        data_dir=DATA_FILE.parent,
        output_dir=output_dir
    )

    # - All classes
    class_thresholds = None
    n_classes = len(np.unique(data_df.loc[:, LABEL].values.flatten()))
    output_dir = ROOT_OUTPUT_DIR / f'outputs_{N_FOLDS}_cv_{n_classes}_classes_{TS}'
    os.makedirs(output_dir, exist_ok=True)
    run_cv(
        data_df=data_df,
        classifier=CLASSIFIER,
        n_folds=N_FOLDS,
        features=FEATURES,
        label=LABEL,
        class_thresholds=class_thresholds,
        data_dir=DATA_FILE.parent,
        output_dir=output_dir
    )
