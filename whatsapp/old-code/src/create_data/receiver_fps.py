import cv2
import numpy as np
import time
import mss
import datetime
import csv
import sys


class FPScalculator:
    def __init__(self, duration=10):
        self.unique_frames_per_second = []
        # Define the region where the number is displayed (replace these coordinates with your own)
        self.x, self.y, self.width, self.height = 985, 535, 10, 10
        self.start_time = None  # update at beginning of 'run' method
        self.grabs = []
        #self.fps_list = np.zeros(duration+1).tolist()
        self.fps_list = []  # list of lists[2]: [end time, number of frames received in the window]
        self.duration = duration
        self.end_time_stamps = []
        self.screenshots_list = []

    def run(self):
        self.start_time = time.time()
        #print(f'receiver fps ver4: run start time: {self.start_time}')
        prev_time = int(self.start_time)
        prev_len = 0
        while True:
            # Capture frames
            with mss.mss() as sct:
                monitor = {"top": self.y, "left": self.x, "width": self.width, "height": self.height}
                screenshot = np.array(sct.grab(monitor))
            self.grabs.append((int(time.time())+1, screenshot))  # (et - end time of window, screenshot)

            if int(time.time()) - prev_time >= 1.0:
                self.screenshots_list.append((int(time.time()), len(self.grabs) - prev_len))
            #    print(f'FPS: overall screen shots in last sec: {len(self.grabs) - prev_len}')
                prev_time = int(time.time())
                prev_len = len(self.grabs)

            # Check if the duration has elapsed
            if time.time() - self.start_time >= self.duration:
                print(f'FPS: overall screen shots: {len(self.grabs)}, during {self.duration} seconds')
                # display al screen grabs that were taken
                #for frame in grabs:
                #    cv2.imshow('Frame', frame)
                #    # Wait for a key press and then close the window
                #    cv2.waitKey(0)
                #    cv2.destroyAllWindows()
                break

    def calculate_fps(self):
        for (end_time_stamp, image) in self.grabs:
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Check if the current frame is different from the previous one
            if len(self.unique_frames_per_second) == 0 or not np.array_equal(gray_frame, self.unique_frames_per_second[-1][1]):
                self.unique_frames_per_second.append((end_time_stamp, gray_frame))

        print(f'FPS: total unique frames captured: {len(self.unique_frames_per_second)}')

        first_end_time_step = self.unique_frames_per_second[0][0]
        # initialize fps_list
        for i in range(self.duration + 1):
            self.fps_list = [[first_end_time_step + i, 0] for i in range(self.duration + 1)]


        for i, (end_time_stamp, _) in enumerate(self.unique_frames_per_second):
            self.fps_list[end_time_stamp - first_end_time_step][1] += 1

            #if i != 0 and end_time_stamp == self.unique_frames_per_second[i - 1][0]:
            #    self.fps_list[end_time_stamp - first_end_time_step][1] += 1
            #else:
            #    self.end_time_stamps.append(end_time_stamp)

        # Print fps list: the number of unique frames displayed each second
        for i, (et, fps) in enumerate(self.fps_list):
            print(f'sec: {i}, end epoch_time: {et}: {fps} FPS')

        # Extract the fps values from self.fps_list
        fps_values = [t[1] for t in self.fps_list]

        # calculate average
        avg = np.array(fps_values[1:self.duration]).mean()
        print(f'\naverage fps for {self.duration} seconds: {avg}')

        # calculate standard deviation
        std = np.array(fps_values[1:self.duration]).std()
        print(f'standard deviation: {std}')

    def display_unique_frames_captured(self):
        print(f'total unique frames captured: {len(self.unique_frames_per_second)}')
        for (_, unique_frame) in self.unique_frames_per_second:
            cv2.imshow('Frame', unique_frame)
            # Wait for a key press and then close the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def display_all_grabs(self):
        print(f'total screen shots captured: {len(self.grabs)}')
        for (_, image) in self.grabs:
            cv2.imshow('Frame', image)
            # Wait for a key press and then close the window
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def write_to_csv(fps_list, screenshots_num_l, dir_path):
    filename = dir_path + "\\fpsLabels.csv"
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['fps', 'et', 'screenshots_num']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (et, fps), (_, screenshots_num) in zip(fps_list, screenshots_num_l):
            writer.writerow({'fps': fps, 'et': et, 'screenshots_num': screenshots_num})

    print("\nfps labels csv file was created")


def write_to_csv_screenshots(list, dir_path):
    filename = dir_path + "\\screenshots_num.csv"
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['screenshots_num', 'et']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for (et, num) in list:
            writer.writerow({'screenshots_num': num, 'et': et})

    print("\nscreenshots_num csv file was created")


def main():
    time.sleep(3)
    prog_start_time = time.time()

    # Check if arguments are provided
    if len(sys.argv) < 4:
        print("Usage: python create_pcap.py <arg1> <arg2>")
        return

    initial_time = sys.argv[1]
    name = sys.argv[2]
    dir_path = sys.argv[3]
    duration = int(sys.argv[4])  # Duration of capture in seconds

    print(f'receiver fps ver4: start time: {prog_start_time}')

    fpsCalculator = FPScalculator(duration)
    fpsCalculator.run()
    print(f'receiver fps ver 4: total duration of captures part: {time.time() - prog_start_time}')

    # second part: calculate bps, create csv file
    time.sleep(5)
    fpsCalculator.calculate_fps()
    write_to_csv(fpsCalculator.fps_list, fpsCalculator.screenshots_list, dir_path)
    print(f'\ntotal duration of receiver fps program: {time.time() - prog_start_time}')
    #fpsCalculator.display_unique_frames_captured()


if __name__ == "__main__":
    main()
