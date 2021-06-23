import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.filters import threshold_otsu
import imutils
import os
import csv
import glob
from plotting import *
from aStar import *


def timer(original_func):
    """ Timer decorator for debugging purposes """
    import time

    def wrapper_function(*args, **kwargs):
        t1 = time.time()
        result = original_func(*args, **kwargs)
        t2 = time.time() - t1
        print(f"{original_func.__name__.upper()} ran within {t2} seconds")
        return result

    return wrapper_function


def get_image(img_path):
    """ Reads and returns a binary image from given path. """
    image = cv2.imread(img_path, 0)
    image = cv2.bitwise_not(image)
    assert len(image.shape) == 2, "Trying to read image while being in a wrong folder, or provided path is wrong."
    return image


def get_binary(img):
    """ Returns a binarized image being undergone 'threshold_otsu()' """
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:  # Fully black or white image
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary * 1
    return binary


def rotate_image(image):
    """
    A function that takes a binary image and rotates it until the lines found by Hough-Transform become
    perpendicular or close to perpendicular to the vertical axis.

    Returns a rotated binary image.
    """

    # tested_angles = np.linspace(np.pi* 49/100, np.pi *51/100, 100)
    tested_angles = np.linspace(-np.pi * 40 / 100, -np.pi * 50 / 100, 100)
    hspace, theta, dist, = hough_line(image, tested_angles)
    h, q, d = hough_line_peaks(hspace, theta, dist)

    angle_list = []  # Create an empty list to capture all angles
    dist_list = []
    origin = np.array((0, image.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(
            hspace, theta, dist, min_distance=50, threshold=0.76 *
                                                            np.max(hspace))):
        # Not for plotting but later calculation of angles
        angle_list.append(angle)
        dist_list.append(dist)
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        # ax[2].plot(origin, (y0, y1), '-r')
    ave_angle = np.mean(angle_list)
    ave_dist = np.mean(dist_list)
    x0, x1 = (ave_dist - origin * np.cos(ave_angle)) / np.sin(ave_angle)

    # Convert angles from radians to degrees (1 rad = 180/pi degrees)
    angles = [a * 180 / np.pi for a in angle_list]
    change = 90 - -1 * np.mean(angles)
    new_image = cv2.bitwise_not(imutils.rotate_bound(image, -change))

    # plotHoughTransform(hspace, theta, dist, x0, x1, origin, new_image)

    # Compute difference between the two lines
    angle_difference = np.max(angles) - np.min(angles)

    return new_image


def calc_outlier(data, method="std"):
    if method == "iqr":
        # method1: interquartile
        q3, q1 = np.percentile(data, [75, 25])
        iqr = q3 - q1
        outlier = q1 - 1.5 * iqr
    else:
        # method2: standard deviation
        outlier = np.mean(data) - np.std(data)
    return outlier


@timer
def get_lines(new_image):
    """
    A function that takes a binary image and searches for lines.

    Returns the top and bottom Y coordinates of the rectangular sections that do not contain characters,
    as well as a number of auxiliary parameters.
    """

    # A) Obtain active pixels in each row (horizontal projection)
    h_hist = []
    row_len = new_image.shape[1]
    for row in new_image:
        h_hist.append(row_len - len(row.nonzero()[0]))

    local_lines = []  # list of pixels in a peak's neighborhood from left to right
    thr_lines = {}  # thresholded lines; containing lines of interest and the horizontal projections thereof
    c = 0  # counter variable
    thr_num = max(h_hist) * 0.09
    for idx, line_sum in enumerate(h_hist):
        if line_sum >= thr_num and h_hist[idx - 1] > thr_num and idx > 0:
            local_lines.append(line_sum)
            c += 1
        # once line_sum is no longer larger than thresh and it has been in the past
        elif len(local_lines) > 0:
            # add local_lines to a dict and start local_lines again
            thr_lines.setdefault(idx - c, local_lines)
            local_lines = []
            c = 0

    # B) Obtain the starting and ending locations, the max value, and the height of each peak's neighborhood (section)
    locations = []
    for idx, line in enumerate(thr_lines.items()):
        y_loc_start = line[0]
        line_projections = line[1]  # line projections within the neighbourhood of the current peak
        height = len(line_projections)
        locations.append([y_loc_start, y_loc_start + height])  # starting and ending location of the section

    # C) Combining sections that are too close together
    # distances between consecutive sections (SEC_n+1_top - SEC_n_bottom)
    distances = [
        locations[sec + 1][0] - locations[sec][1]
        for sec in range(len(locations) - 1)
    ]
    min_distance = calc_outlier(distances) if calc_outlier(distances) > 18 else 18
    # print(distances, '\n', min_distance)

    locations_new = []
    idx = 0
    while idx < len(locations):  # Run combination algorithm from top to bottom
        if idx < len(distances):  # len(distances) = len(locations)-1
            distance = distances[idx]
            locations_new.append(locations[idx])
        else:  # End of algorithm
            if locations[idx][1] - locations[idx][0] > min_distance:
                locations_new.append(locations[idx])
            break

        idx2 = 1
        while distance < min_distance:
            locations_new[-1][1] = locations[idx + idx2][1]  # merge current location with next
            if idx + idx2 < len(distances):
                distance = distances[idx + idx2]  # merge current distance with next
            else:
                break
            idx2 += 1
        idx += idx2

    # D) Adding buffer, based on average height of the NEW (!) sections, to each section that is too small
    section_heights_new = [x[1] - x[0] for x in locations_new]
    avg_sh_new = np.mean(section_heights_new)
    for idx, loc in enumerate(locations_new):
        if section_heights_new[idx] < avg_sh_new:
            for i in range(len(loc)):
                if idx == 0:
                    # top lines are pushed up
                    locations_new[idx][i] -= int(avg_sh_new / 5)
                else:
                    # bottom lines are pushed down
                    locations_new[idx][i] += int(avg_sh_new / 6)

    # E) Remove sections with too small height (may occur due to very short lines with scattered artefacts)
    locations_final = []
    min_height = int(calc_outlier(section_heights_new) / 3) + 2
    for idx in range(len(section_heights_new)):
        if section_heights_new[idx] >= min_height:
            locations_final.append(locations_new[idx])
    section_heights_final = [x[1] - x[0] for x in locations_final]
    avg_sh_final = np.mean(section_heights_final)

    # F) Obtaining the locations of the inbetween (non-character) sections
    # and adding a bonus line to the top of the image
    mid_lines = []
    top_line = [
        locations_final[0][0] - int(avg_sh_final * 2.5),
        locations_final[0][0] - int(avg_sh_final)
    ]
    mid_lines.append(top_line)

    # TODO: This might not even be necessary at all
    for sec in range(len(locations_final) - 1):
        if locations_final[sec][0] < locations_final[sec][1]:
            beginning = locations_final[sec][1]  # bottom line of peak_n
            end = locations_final[sec + 1][0]  # top line of peak_n+1
            mid_lines.append([beginning, end])

    # sanity check
    mid2 = []
    for sec in mid_lines:
        if sec[0] < sec[1]:
            mid2.append(sec)

    # Adding a bonus line to the bottom of the image
    bottom_line = [
        locations_final[-1][1] + int(avg_sh_final),
        locations_final[-1][1] + int(avg_sh_final * 2)
    ]
    mid2.append(bottom_line)

    return mid2, avg_sh_final, h_hist, thr_num,


def extract_char_section_from_image(image, upper_line, lower_line):
    """ Returns a rectangular section that contain characters using the original image and two consecutive lines from
    top to bottom """
    upper_boundary = np.min(upper_line[:, 0])
    lower_boundary = np.max(lower_line[:, 0])
    img_copy = np.copy(image)
    r, c = img_copy.shape
    x_axis = 0
    y_axis = 1
    for step in upper_line:
        img_copy[0:step[x_axis], step[y_axis]] = 0
    for step in lower_line:
        img_copy[step[x_axis]:r, step[y_axis]] = 0
    return img_copy[upper_boundary:lower_boundary, :]


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


def line_segmentation(img_path, new_folder_path):
    image = get_image(img_path)
    image = rotate_image(image)
    binary_image = get_binary(image)

    if not os.path.exists(new_folder_path):
        print("Running line segmentation on new image...")
        os.makedirs(new_folder_path)
        # run image-processing
        mid_lines, avg_lh, hist, thr_num = get_lines(image)
        plot_hist(hist,
                  thr_num,
                  save=True,
                  folder_path=new_folder_path,
                  overwrite_path=False)
        plot_hist_lines_on_image(binary_image,
                                 mid_lines,
                                 save=True,
                                 folder_path=new_folder_path,
                                 overwrite_path=False)
        # Find paths with A*
        paths = find_paths(mid_lines, binary_image, avg_lh)
        plot_paths_next_to_image(binary_image,
                                 paths,
                                 save=True,
                                 folder_path=new_folder_path,
                                 overwrite_path=False)
        # save paths
        for idx, path in enumerate(paths):
            save_path(path, f"{new_folder_path}/path_{idx}.csv")
        paths = [np.array(path) for path in paths]
    else:
        # load paths
        file_paths_list = sorted(glob.glob(f'{new_folder_path}/*.csv'),
                                 key=get_key)
        paths = []  # a* paths
        for file_path in file_paths_list:
            line_path = load_path(file_path)
            paths.append(line_path)

    section_images = []
    line_count = len(paths)
    for line_index in range(line_count -
                            1):  # |-------- extract sections from loaded paths
        section_image = extract_char_section_from_image(binary_image, paths[line_index],
                                                        paths[line_index + 1])
        section_images.append(section_image)
    return section_images
