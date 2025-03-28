import os
import pandas as pd

def convertRtp(x):
    #tshark_path = "C:\\Program Files\\Wireshark\\"
    for t in os.listdir(x):
        if t.endswith('.pcap'):
            output_csv = f"{x}/{t[:-5]}.csv"
            print(f'convert: {t} -> {t[:-5]}Rtp.csv')
            tshark_cmd = f"""
            tshark -r {x}/{t} -d udp.port==1024-49152,rtp -t e -T fields \
            -e frame.time_relative -e frame.time_epoch -e ip.src -e ip.dst -e ip.proto \
            -e ip.len -e udp.srcport -e udp.dstport -e udp.length -e rtp.ssrc -e rtp.timestamp \
            -e rtp.seq -e rtp.p_type -e rtp.marker -E separator=, -E header=y > {x}/{t[:-5]}Rtp.csv
            """
            os.system(tshark_cmd)
            print(f"{t[:-5]}.csv was created successfully")


if __name__ == "__main__":
    father_dir = "C:\\final_project\pcap_files"

    for f in os.listdir(father_dir):
        convertRtp(father_dir+f'\{f}')