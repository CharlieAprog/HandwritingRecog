import os
import csv
import numpy as np
import glob
from plotting import *
from lineSegmentation import *
from wordSegmentation import *
from characterSegmentation import *


def save_path(path, file_name):
    """ Saves a numpy array into csv format for a SINGLE path """
    with open(file_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for row in path:
            csvwriter.writerow(row)


def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split('_')[1]
    return int(int_part)


def load_path(file_name):
    """ Loads a csv formatted path file into a numpy array. """
    return np.loadtxt(file_name, delimiter=',', dtype=int)


# |----------------------------------------------|
# |    Global variables used by other files      |
# |----------------------------------------------|
image_num = 4
img_path = f'data/image-data/binaryRenamed/{image_num}.jpg'
new_folder_path = f"data/image-data/paths/{os.path.basename(img_path).split('.')[0]}"

if __name__ == "__main__":

    # |--------------------------------------------|
    # |            LINE SEGMENTATION               |
    # |--------------------------------------------|
    image = getImage(img_path)
    image = rotateImage(image)
    binary_image = get_binary(image)

    if not os.path.exists(
            new_folder_path):  #|-------- paths for image do not exist
        os.makedirs(new_folder_path)
        # run image-processing
        mid_lines, top_line, bottom_line, avg_lh, hist, thr_num = getLines(
            image)
        plotHist(hist,
                 thr_num,
                 save=True,
                 folder_path=new_folder_path,
                 overwrite_path=False)
        plotHistLinesOnImage(binary_image,
                             mid_lines,
                             save=True,
                             folder_path=new_folder_path,
                             overwrite_path=False)
        paths = find_paths(mid_lines, binary_image, avg_lh)
        plotPathsNextToImage(binary_image,
                             paths,
                             save=True,
                             folder_path=new_folder_path,
                             overwrite_path=False)
        # save paths
        for idx, path in enumerate(paths):
            save_path(path, f"{new_folder_path}/path_{idx}.csv")
    else:  #|-------- paths for image exist
        # load paths
        file_paths_list = sorted(glob.glob(f'{new_folder_path}/*.csv'),
                                 key=get_key)
        paths = []  # a* paths
        sections_loaded = []
        for file_path in file_paths_list:
            line_path = load_path(file_path)
            paths.append(line_path)

    # |--------------------------------------------|
    # |            WORD SEGMENTATION               |
    # |--------------------------------------------|
    line_images = []
    line_count = len(paths)
    for line_index in range(line_count -
                            1):  #|-------- extract lines from loaded paths
        line_image = extract_line_from_image(binary_image, paths[line_index],
                                             paths[line_index + 1])
        line_images.append(line_image)

    words_in_lines = []  #|-------- pad obtained lines
    sliding_words_in_line = []
    for line_num in range(len(line_images)):
        line = trim_line(line_images[line_num])
        if line.shape[0] == 0 or line.shape[1] == 0:
            continue
        vertical_projection = np.sum(line, axis=0)

        # plot the vertical projects
        fig, ax = plt.subplots(nrows=2, figsize=(10, 5))
        plt.xlim(0, line.shape[1])
        ax[0].imshow(line, cmap="gray")
        ax[1].plot(vertical_projection)
        plt.show()

        # we will go through the vertical projections and
        # find the sequence of consecutive white spaces in the image

        dividers = segment_words(line, vertical_projection)
        words = []
        sliding_words = []
        box_width = 70
        shift = 20
        print(line.shape)
        for window in dividers:
            word = line[:, window[0]:window[1]]
            trimmed_word = trim_line(np.rot90(trim_line(np.rot90(word).astype(int)),axes=(1, 0)).astype(int))
            sliding_words.append(slide_over_word(trimmed_word, box_width, shift))
            #plotSimpleImages(sliding_words[-1])
            words.append(trimmed_word)
        words_in_lines.append(words)
        plotGrid([line, vertical_projection].extend(words))
        # plotWordsInLine(line, words)

    # |--------------------------------------------|
    # |        CHARACTER SEGMENTATION              |
    # |--------------------------------------------|
    image = cv2.imread("characterSegmentation_TestImage.jpeg", 0)
    image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels = cv2.connectedComponents(image)
    clusters = getComponentClusters(num_labels, labels)
    box_boundaries = getBoundingBoxBoundaries(image, clusters)

    plotConnectedComponentBoundingBoxes(image, box_boundaries)
