import subprocess
import os
from brisque import BRISQUE
import numpy as np
from PIL import Image  # PIL is used to read the image file
import matplotlib.pyplot as plt
import time
import csv
import shutil

framerate = 1


def calculate_avg_brisque(image_folder):
    print("Calculate brisque scores...")
    images = [im for im in os.listdir(image_folder) if im.endswith('.png')]
    brisque_model = BRISQUE(url=False)
    avg_brisque_scores = []

    for i in range(0, len(images), framerate):
        batch_images = images[i:i + framerate]
        batch_scores = []

        for im in batch_images:
            image_path = image_folder + "\\" + im
            image = Image.open(image_path)
            image_array = np.array(image)

            score = brisque_model.score(image_array)
            batch_scores.append(score)

        average_score = np.mean(batch_scores)
        avg_brisque_scores.append(round(average_score, 5))

    print("Finished calculate brisque scores")
    return avg_brisque_scores


def write_et_to_csv(et_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['et']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for et in et_list:
            writer.writerow({'et': et})

    print("brisque labels csv file - contains only et was created")


def write_to_csv(scores, et_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['brisque', 'et']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for score, et in zip(scores, et_list):
            writer.writerow({'brisque': score, 'et': et})

    print("brisque labels csv file was created")


def add_column_to_csv(new_col_data, new_col_header, filename, new_filename):
    with open(filename, 'r', newline='') as csvfile, \
            open(new_filename, 'w', newline='') as new_csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + [new_col_header]  # Add the new column header

        writer = csv.DictWriter(new_csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            row[new_col_header] = new_col_data.pop(0)  # Add data from the new column
            writer.writerow(row)

    # Replace the original file with the new one
    shutil.move(new_filename, filename)


def create_brisque_file_all_dirs(main_folder):
    tuples_list = []

    # Iterate over all folders in the main folder
    for i, folder_name in enumerate(os.listdir(main_folder)):
        folder_path = os.path.join(main_folder, folder_name)

        # Check if the item is a directory
        if os.path.isdir(folder_path):
            print(f'dir number {i + 1} - {folder_path}')
            # Find images dir - 'ffmpeg_images'
            for dirpath, dirnames, filenames in os.walk(folder_path):
                if 'ffmpeg_images' in dirnames:
                    images_folder = os.path.join(dirpath, 'ffmpeg_images')
                    # calculates brisque scores list of all images
                    average_scores = calculate_avg_brisque(images_folder)
                    for j, score in enumerate(average_scores):
                        print(f'image {j+1}: score: {score}')
                    # add brisque score col for the csv which contains only the related et
                    if os.path.exists(folder_path+'\\brisqueLabels.csv'):
                        add_column_to_csv(average_scores, 'brisque', folder_path+'\\brisqueLabels.csv', 'tempFile.csv')
                    else:
                        print('no brisqueLabels.csv file found')
                else:
                    print('no ffmpeg_images dir found')
                break


if __name__ == "__main__":

    main_folder = "C:\\final_project\pcap_files"
    create_brisque_file_all_dirs(main_folder)


