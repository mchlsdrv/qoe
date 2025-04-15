import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


FEATURES_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/output/features.csv')
FEATURES_FILE.is_file()


feats_df = pd.read_csv(FEATURES_FILE)
feats_df.shape
feats_df.head()

PCAP_FILE = '/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/data/bandwidth/2024_06_16_12_56_250KBps/pcap_2024_06_16_12_56_250KBps.csv'
pcap_df = pd.read_csv(PCAP_FILE)
pcap_df = pcap_df.astype({'frame.time_epoch': int})
pcap_df.head()

pcap_gb = pcap_df.groupby('frame.time_epoch').apply(list)
pcap_gb
