import os
import pathlib
import numpy as np
import pandas as pd
import pcapfile as pf
import dpkt
import datetime
from dpkt.utils import mac_to_str, inet_to_str
from scapy.all import *
from tqdm import tqdm

# Non-VPN Data
print(f'*********************')
print(f'Non-VPN Traffic')
print(f'*********************')
PCAP_FILE = pathlib.Path('/Users/mchlsdrv/Desktop/projects/phd/qoe/whatsapp/data/bandwidth/2024_06_16_12_56_250KBps/pcap_2024_06_16_12_56_250KBps.pcap')
pcap = rdpcap(str(PCAP_FILE))
non_vpn_pckt_lens = np.array([])
non_vpn_pckt_types = [0, 0, 0]
non_vpn_arrival_times = np.array([])
for pckt in tqdm(pcap):
    try:
        non_vpn_pckt_lens = np.append(non_vpn_pckt_lens, pckt.len)
        non_vpn_arrival_times = np.append(non_vpn_arrival_times, float(pckt.time))
    except AttributeError as err:
        pass

    if pckt.name == 'Ethernet':
        non_vpn_pckt_types[0] += 1
    elif pckt.name == 'IP':
        non_vpn_pckt_types[1] += 1
    else:
        non_vpn_pckt_types[2] += 1

non_vpn_arrival_times -= non_vpn_arrival_times[0]
non_vpn_time_deltas = non_vpn_arrival_times[1:] - non_vpn_arrival_times[:-1]
non_vpn_pckt_len_mu, non_vpn_pckt_len_std = non_vpn_pckt_lens.mean(), non_vpn_pckt_lens.std()
print(f'Mean packet length (Non-VPN): {non_vpn_pckt_len_mu:.2f}+/-{non_vpn_pckt_len_std:.3f}')

