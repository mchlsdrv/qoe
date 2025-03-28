import os
import shutil


def convert(x):
    tshark_path = "C:\\Program Files\\Wireshark\\"

    for t in os.listdir(x):
        if t.endswith('.pcap'):
            output_csv = f"{x}/{t[:-5]}.csv"
            print(f'convert: {t} -> {t[:-5]}.csv')
            tshark_cmd = f'"{tshark_path}tshark" -r {x}/{t} -Y udp -T fields -e frame.time_relative -e frame.time_epoch -e ip.src -e ip.dst -e ip.proto -e ip.len -e udp.srcport -e udp.dstport -e udp.length -E separator=, -E header=y > {output_csv}'
            os.system(tshark_cmd)
            print(f"{t[:-5]}.csv was created successfully")


if __name__ == "__main__":

    pcap_dir_path = "C:\\final_project\pcap_files"
    convert(pcap_dir_path)
