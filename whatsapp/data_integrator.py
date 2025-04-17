import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


DATA_ROOT = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/data')
DATA_ROOT.is_dir()

dataset = pd.DataFrame()
for root, sub_dirs, files in os.walk(DATA_ROOT):
    for limit_sub_dir in sub_dirs:
        limit_sub_dir_path = pathlib.Path(f"{root}/{limit_sub_dir}")
        exp_dirs = os.listdir(limit_sub_dir_path)
        for exp_dir in exp_dirs:
           files = os.listdir(limit_sub_dir_path / exp_dir)
           for file in files:
               data = pd.read_csv(limit_sub_dir_path / exp_dir / file)
               print(files)