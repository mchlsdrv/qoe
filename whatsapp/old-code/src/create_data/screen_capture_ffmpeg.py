import subprocess
import time
import sys
import os
from images2brisque import calculate_avg_brisque, write_to_csv, write_et_to_csv


def capture_screen(output_folder, num_frames, framerate):
    print("begin with ffmpeg screen captures...")
    X_OFFSET = '0'
    Y_OFFSET = '0'
    WIDTHxHEIGHT = '1920x1080'
    ffmpeg_path = r'C:\final_project\ffmpeg.exe'
    capture_command = f'{ffmpeg_path} -f gdigrab -framerate {framerate} -offset_x {X_OFFSET} -offset_y {Y_OFFSET} -video_size {WIDTHxHEIGHT} -i desktop -q:v 0 -frames {num_frames} {output_folder}\\output_%03d.png'
    process = subprocess.Popen(capture_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()

    if process.returncode == 0:
        print("screen capture ffmpeg: Screen capture completed successfully.")
    else:
        print(f"screen capture ffmpeg: Error during screen capture:\n{err.decode()}")


if __name__ == "__main__":
    framerate = 1
    #timeout = 60

    # Check if arguments are provided
    if len(sys.argv) < 4:
        print("screen capture ffmpeg: screen_capture_ffmpeg.py didn't get all arguments")

    initial_time = sys.argv[1]
    name = sys.argv[2]
    dir_path = sys.argv[3]
    timeout = int(sys.argv[4])  # Duration of capture in seconds

    # Choose the folder for capturing images - create dir if it doesn't exist
    capture_folder = dir_path + "\\ffmpeg_images"
    if not os.path.exists(capture_folder):
        os.makedirs(capture_folder)

    start_time = time.time()
    print(f'screen capture ffmpeg: start time: {start_time}')
    # from measurement, takes approximately 0.7 second until ffmpeg starts to capture.
    # update start time to start time of first ffmpeg screen capture
    start_time += 0.7
    # Run screen capture command
    capture_screen(capture_folder, str(timeout * framerate), str(framerate))
    print(f'screen capture ffmpeg: end time of captures part: {time.time()}')
    print(f'screen capture ffmpeg: total duration of captures part: {time.time() - start_time}')

    # ----- Create brisque labels csv file -----
    time.sleep(5)
    output_csv = dir_path + '\\brisqueLabels.csv'
    et_list = [int(start_time) + i for i in range(1, timeout + 1)]

    # calculate brisque average scores per time slot and create csv labels file:
    #average_scores = calculate_avg_brisque(capture_folder)
    #write_to_csv(average_scores, et_list, output_csv)

    # alternatively, create csv contains only the time stamps, remaining to calculate brisque average scores later.
    write_et_to_csv(et_list, output_csv)
