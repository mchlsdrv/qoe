import numpy as np
import cv2
from scipy.special import gamma
import sys
import os
import csv
import pandas as pd
import shutil

framerate = 1


def calculate_mscn(dis_image):
    dis_image = dis_image.astype(np.float32)  # 类型转换十分重要
    ux = cv2.GaussianBlur(dis_image, (7, 7), 7/6)
    ux_sq = ux*ux
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(dis_image**2, (7, 7), 7/6)-ux_sq))

    mscn = (dis_image-ux)/(1+sigma)

    return mscn

# Function to segment block edges


def segmentEdge(blockEdge, nSegments, blockSize, windowSize):
    # Segment is defined as a collection of 6 contiguous pixels in a block edge
    segments = np.zeros((nSegments, windowSize))
    for i in range(nSegments):
        segments[i, :] = blockEdge[i:windowSize]
        if(windowSize <= (blockSize+1)):
            windowSize = windowSize+1

    return segments


def noticeDistCriterion(Block, nSegments, blockSize, windowSize, blockImpairedThreshold, N):
    # Top edge of block
    topEdge = Block[0, :]
    segTopEdge = segmentEdge(topEdge, nSegments, blockSize, windowSize)

    # Right side edge of block
    rightSideEdge = Block[:, N-1]
    rightSideEdge = np.transpose(rightSideEdge)
    segRightSideEdge = segmentEdge(
        rightSideEdge, nSegments, blockSize, windowSize)

    # Down side edge of block
    downSideEdge = Block[N-1, :]
    segDownSideEdge = segmentEdge(
        downSideEdge, nSegments, blockSize, windowSize)

    # Left side edge of block
    leftSideEdge = Block[:, 0]
    leftSideEdge = np.transpose(leftSideEdge)
    segLeftSideEdge = segmentEdge(
        leftSideEdge, nSegments, blockSize, windowSize)

    # Compute standard deviation of segments in left, right, top and down side edges of a block
    segTopEdge_stdDev = np.std(segTopEdge, axis=1)
    segRightSideEdge_stdDev = np.std(segRightSideEdge, axis=1)
    segDownSideEdge_stdDev = np.std(segDownSideEdge, axis=1)
    segLeftSideEdge_stdDev = np.std(segLeftSideEdge, axis=1)

    # Check for segment in block exhibits impairedness, if the standard deviation of the segment is less than blockImpairedThreshold.
    blockImpaired = 0
    for segIndex in range(segTopEdge.shape[0]):
        if((segTopEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segRightSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segDownSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segLeftSideEdge_stdDev[segIndex] < blockImpairedThreshold)):
            blockImpaired = 1
            break

    return blockImpaired


def noiseCriterion(Block, blockSize, blockVar):
    # Compute block standard deviation[h,w,c]=size(I)
    blockSigma = np.sqrt(blockVar)
    # Compute ratio of center and surround standard deviation
    cenSurDev = centerSurDev(Block, blockSize)
    # Relation between center-surround deviation and the block standard deviation
    blockBeta = (abs(blockSigma-cenSurDev))/(max(blockSigma, cenSurDev))

    return blockSigma, blockBeta

# Function to compute center surround Deviation of a block


def centerSurDev(Block, blockSize):
    # block center
    center1 = int((blockSize+1)/2)-1
    center2 = center1+1
    center = np.vstack((Block[:, center1], Block[:, center2]))
    # block surround
    Block = np.delete(Block, center1, axis=1)
    Block = np.delete(Block, center1, axis=1)

    # Compute standard deviation of block center and block surround
    center_std = np.std(center)
    surround_std = np.std(Block)

    # Ratio of center and surround standard deviation
    cenSurDev = (center_std/surround_std)

    # Check for nan's
    # if(isnan(cenSurDev)):
    #     cenSurDev = 0

    return cenSurDev

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

def piqe(im):
    blockSize = 16  # Considered 16x16 block size for overall analysis
    activityThreshold = 0.1  # Threshold used to identify high spatially prominent blocks
    blockImpairedThreshold = 0.1  # Threshold identify blocks having noticeable artifacts
    windowSize = 6  # Considered segment size in a block edge.
    nSegments = blockSize-windowSize+1  # Number of segments for each block edge
    distBlockScores = 0  # Accumulation of distorted block scores
    NHSA = 0  # Number of high spatial active blocks.

    # pad if size is not divisible by blockSize
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    originalSize = im.shape
    rows, columns = originalSize
    rowsPad = rows % blockSize
    columnsPad = columns % blockSize
    isPadded = False
    if(rowsPad > 0 or columnsPad > 0):
        if rowsPad > 0:
            rowsPad = blockSize-rowsPad
        if columnsPad > 0:
            columnsPad = blockSize-columnsPad
        isPadded = True
        padSize = [rowsPad, columnsPad]
    im = np.pad(im, ((0, rowsPad), (0, columnsPad)), 'edge')

    # Normalize image to zero mean and ~unit std
    # used circularly-symmetric Gaussian weighting function sampled out
    # to 3 standard deviations.
    imnorm = calculate_mscn(im)

    # Preallocation for masks
    NoticeableArtifactsMask = np.zeros(imnorm.shape)
    NoiseMask = np.zeros(imnorm.shape)
    ActivityMask = np.zeros(imnorm.shape)

    # Start of block by block processing
    total_var = []
    total_bscore = []
    total_ndc = []
    total_nc = []

    BlockScores = []
    for i in np.arange(0, imnorm.shape[0]-1, blockSize):
        for j in np.arange(0, imnorm.shape[1]-1, blockSize):
             # Weights Initialization
            WNDC = 0
            WNC = 0

            # Compute block variance
            Block = imnorm[i:i+blockSize, j:j+blockSize]
            blockVar = np.var(Block)

            if(blockVar > activityThreshold):
                ActivityMask[i:i+blockSize, j:j+blockSize] = 1
                NHSA = NHSA+1

                # Analyze Block for noticeable artifacts
                blockImpaired = noticeDistCriterion(
                    Block, nSegments, blockSize-1, windowSize, blockImpairedThreshold, blockSize)

                if(blockImpaired):
                    WNDC = 1
                    NoticeableArtifactsMask[i:i +
                                            blockSize, j:j+blockSize] = blockVar

                # Analyze Block for guassian noise distortions
                [blockSigma, blockBeta] = noiseCriterion(
                    Block, blockSize-1, blockVar)

                if((blockSigma > 2*blockBeta)):
                    WNC = 1
                    NoiseMask[i:i+blockSize, j:j+blockSize] = blockVar

                # Pooling/ distortion assigment
                # distBlockScores = distBlockScores + \
                #     WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2)

                if WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2) > 0:
                    BlockScores.append(
                        WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2))

                total_var = [total_var, blockVar]
                total_bscore = [total_bscore, WNDC *
                                (1-blockVar) + WNC*(blockVar)]
                total_ndc = [total_ndc, WNDC]
                total_nc = [total_nc, WNC]

    BlockScores = sorted(BlockScores)
    lowSum = sum(BlockScores[:int(0.1*len(BlockScores))])
    Sum = sum(BlockScores)
    Scores = [(s*10*lowSum)/Sum for s in BlockScores]
    C = 1
    Score = ((sum(Scores) + C)/(C + NHSA))*100

    # if input image is padded then remove those portions from ActivityMask,
    # NoticeableArtifactsMask and NoiseMask and ensure that size of these masks
    # are always M-by-N.
    if(isPadded):
        NoticeableArtifactsMask = NoticeableArtifactsMask[0:originalSize[0],
                                                          0:originalSize[1]]
        NoiseMask = NoiseMask[0:originalSize[0], 0:originalSize[1]]
        ActivityMask = ActivityMask[0:originalSize[0], 1:originalSize[1]]

    return Score, NoticeableArtifactsMask, NoiseMask, ActivityMask


def calculate_avg_piqe(image_folder):
    print("Calculate brisque scores...")
    images = [im for im in os.listdir(image_folder) if im.endswith('.png')]
    avg_brisque_scores = []

    for i in range(0, len(images), framerate):
        batch_images = images[i:i + framerate]
        batch_scores = []

        for im in batch_images:
            image_path = image_folder + "\\" + im
            im_read = cv2.imread(image_path)
            score, NoticeableArtifactsMask, NoiseMask, ActivityMask = piqe(im_read)
            print("{}-----piqe score:{}".format(image_path, score))
            batch_scores.append(score)

        average_score = np.mean(batch_scores)
        avg_brisque_scores.append(round(average_score, 5))

    print("Finished calculate piqe scores")
    return avg_brisque_scores

def create_piqe_file_all_dirs(main_folder):
    tuples_list = []

    # Iterate over all folders in the main folder
    #for dataset_name in os.listdir(main_folder):
    #    dataset_path = os.path.join(main_folder, dataset_name)

        # Iterate over all folders in the dataset folder
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
                    average_scores = calculate_avg_piqe(images_folder)
                    for j, score in enumerate(average_scores):
                        print(f'image {j+1}: score: {score}')
                    # add brisque score col for the csv which contains only the related et

                    brisque_csv_path = folder_path + '\\brisqueLabels.csv'
                    brisque_piqe_csv_path = folder_path + '\\piqeLabels.csv'
                    if os.path.exists(brisque_csv_path):


                        # Read the existing CSV file
                        df = pd.read_csv(brisque_csv_path)

                        # Add the new column to the DataFrame
                        df['piqe'] = average_scores

                        # Save the modified DataFrame to a new CSV file
                        df.to_csv(brisque_piqe_csv_path, index=False)
                        #add_column_to_csv(average_scores, 'piqe', folder_path+'\\brisqueLabels.csv', folder_path+ '\\brisque_piqeLabels.csv')
                    else:
                        print('no brisqueLabels.csv file found')
                else:
                    print('no ffmpeg_images dir found')
                break


if __name__ == "__main__":
    '''
    test conventional blindly image quality assessment methods(brisque/niqe/piqe)
    '''

    main_folder = "C:\\final_project\pcap_files"
    create_piqe_file_all_dirs(main_folder)
