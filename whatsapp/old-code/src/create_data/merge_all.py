import multiprocessing
import subprocess
import time
import datetime
import os


def run_file(args):
    filename, *arguments = args
    subprocess.call(['python', filename, *arguments])


if __name__ == '__main__':
    initial_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    name = "validation"
    directory = f"C:\\final_project\pcap_files\\{initial_time}_{name}"
    duration = "5"
    if not os.path.exists(directory):
        os.makedirs(directory)

    files = [
        ('create_pcap.py', initial_time, name, directory, duration),
        ('receiver_fps.py', initial_time, name, directory, duration),
        ('screen_capture_ffmpeg.py', initial_time, name, directory, duration)
    ]

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=len(files))

    # Run the files in parallel
    print(f"start run pcap, ffmpeg captures, Bps calculate files in parallel - start time: {time.time()}")
    pool.map(run_file, files)

    # Close the pool
    pool.close()
    pool.join()
