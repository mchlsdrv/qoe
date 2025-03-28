import os
import datetime
import sys
from pcap2csv import convert
from pcap2csvRtp import convertRtp
import time

#initial_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")


def capture_and_save_pcap(output_path, interface, timeout, initial_time, name):
    tshark_path = "C:\\Program Files\\Wireshark\\"
    duration = str(timeout)

    # Create a new pcap file
    tshark_cmd = f'"{tshark_path}tshark" -i {interface} -a duration:{duration} -w {output_path}\\pcap_{initial_time}_{name}.pcap'
    os.system(tshark_cmd)


def main():
    time.sleep(3)
    prog_start_time = time.time()
    # Check if arguments are provided
    if len(sys.argv) < 4:
        print("Create pcap: create_pcap.py didn't get all arguments")
        return

    initial_time = sys.argv[1]
    name = sys.argv[2]
    dir_path = sys.argv[3]
    timeout = int(sys.argv[4]) + 5
    wifi_interface = "\\Device\\NPF_{CB405C34-9E8D-4DDA-876E-D44BC4CA0E3F}"

    print(f'create pcap: start time: {prog_start_time}')

    capture_and_save_pcap(dir_path, wifi_interface, timeout, initial_time, name)
    print(f'create pcap: total duration of captures part: {time.time() - prog_start_time}')

    # ----- Create pcap csv file -----
    time.sleep(5)
    convertRtp(dir_path)

if __name__ == "__main__":
    main()
