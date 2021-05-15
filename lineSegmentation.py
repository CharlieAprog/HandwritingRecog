from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.util import invert
from skimage.filters import threshold_otsu
from skimage.filters import sobel

import numpy as np
import PIL
from PIL import Image
import PIL.ImageOps
from tqdm import tqdm
import imutils
import cv2
from matplotlib import cm
from matplotlib import pyplot as plt
import pprint
from heapq import *

img_path = 'data/image-data/binaryRenamed/11.jpg'


# image = Image.open(img_path)
# image = PIL.ImageOps.invert(image)
def rotateImage(img_path):
    image = cv2.imread(img_path, 0)
    image = cv2.bitwise_not(image)

    # tested_angles = np.linspace(np.pi* 49/100, np.pi *51/100, 100)
    tested_angles = np.linspace(np.pi * 45 / 100, -np.pi * 55 / 100, 100)
    hspace, theta, dist, = hough_line(image, tested_angles)

    h, q, d = hough_line_peaks(hspace, theta, dist)

    #################################################################
    # Example code from skimage documentation to plot the detected lines
    angle_list = []  # Create an empty list to capture all angles
    dist_list = []
    # Generating figure 1
    # fig, axes = plt.subplots(1, 4, figsize=(15, 6))
    # ax = axes.ravel()
    # ax[0].imshow(image, cmap='gray')
    # ax[0].set_title('Input image')
    # ax[0].set_axis_off()
    # ax[1].imshow(np.log(1 + hspace),
    #              extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
    #              cmap='gray', aspect=1/1.5)
    # ax[1].set_title('Hough transform')
    # ax[1].set_xlabel('Angles (degrees)')
    # ax[1].set_ylabel('Distance (pixels)')
    # ax[1].axis('image')
    # ax[2].imshow(image, cmap='gray')
    origin = np.array((0, image.shape[1]))

    for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist, min_distance=50, threshold=0.72 * np.max(hspace))):
        angle_list.append(angle)  # Not for plotting but later calculation of angles
        dist_list.append(dist)
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        # ax[2].plot(origin, (y0, y1), '-r')

    ave_angle = np.mean(angle_list)
    ave_dist = np.mean(dist_list)

    x0, x1 = (ave_dist - origin * np.cos(ave_angle)) / np.sin(ave_angle)

    # ax[2].plot(origin, (x0, x1), '-b')
    # ax[2].set_xlim(origin)
    # ax[2].set_ylim((image.shape[0], 0))
    # ax[2].set_axis_off()
    # ax[2].set_title('Detected lines')

    ###############################################################
    # Convert angles from radians to degrees (1 rad = 180/pi degrees)
    angles = [a * 180 / np.pi for a in angle_list]
    change = 90 - -1 * np.mean(angles)
    newImage = cv2.bitwise_not(imutils.rotate_bound(image, -change))
    # ax[3].imshow(newImage, cmap='gray')
    # plt.tight_layout()
    # plt.show()
    # Compute difference between the two lines
    angle_difference = np.max(angles) - np.min(angles)
    print(180 - angle_difference)  # Subtracting from 180 to show it as the small angle between two lines

    return newImage


# ------------------------- Histogram part -------------------------
def getLines(newImage):
    hist = []
    row_len = newImage.shape[1]
    for row in newImage:
        hist.append(row_len - len(row.nonzero()[0]))

    temp = []  # list of pixels in a peak's neighborhood from left to right
    thr = {}  # dictionary containing lines of interest
    c = 0  # counter variable
    thr_num = max(hist) * 0.1
    for col, p in enumerate(hist):  # if pixel is above thresh, add it to temp
        if p >= thr_num and hist[col - 1] > thr_num and col > 0:
            temp.append(p)
            c += 1
        elif len(temp) > 0:  # once p is nolonger larger than thresh and it has been in the past
            thr.setdefault(col - c, temp)  # add temp to a dict and start temp again
            temp = []
            c = 0

    line_heights = []
    thr_peaks = []
    for idx, p in enumerate(thr.items()):
        line_heights.append(len(p[1]))
        thr_peaks.append(
            {"loc": [p[0], p[0] + len(p[1])],
             "value": max(p[1]),
             "lh": p[0] + len(p[1]) - p[0]}
        )

    # Combining lines that are too close together
    # -------------------------------------------------
    # Upper and lower Y coordinates of a section that contains characters
    locations = [x['loc'] for x in thr_peaks]
    # Distances between two subsequent sections that contain characters, measured from top to bottom
    distances = [locations[sec + 1][0] - locations[sec][1] for sec in range(len(locations) - 1)]
    min_distance = calc_outlier(distances)
    # Upper and lower Y coordinates of a line that is the result of merging a NON-OUTLIER line with 1 or more
    # OUTLIER lines
    locations_new = []
    idx = 0
    SECOND = 1  # Bottom line's idx within a section that contains characters
    while idx < len(locations):
        if idx < len(distances):
            distance = distances[idx]
            locations_new.append(locations[idx])
        else:
            if locations[idx][1] - locations[idx][0] > min_distance:
                locations_new.append(locations[idx])
            break

        idx2 = 1
        while distance < min_distance:
            locations_new[-1][SECOND] = locations[idx + idx2][SECOND]
            if idx + idx2 < len(distances):
                distance = distances[idx + idx2]
            else:
                break
            idx2 += 1
        idx += idx2
    # -------------------------------------------------

    # Adding buffer, based on average height of the NEW (!) lines, to each line that is too small
    line_heights_new = [x[1]-x[0] for x in locations_new]
    avg_lh = np.mean(line_heights_new)
    for idx, loc in enumerate(locations_new):
        if line_heights_new[idx] < avg_lh:
            for i in range(len(loc)):
                if i == 0:
                    locations_new[idx][i] -= avg_lh // 6  # top lines are pushed up
                else:
                    locations_new[idx][i] += avg_lh // 6   # bottom lines are pushed down

    # obtaining the locations of the INBETWEEN (ideally empty) sections
    mid_lines = []
    for sec in range(len(locations_new) - 1):
        beginning = locations_new[sec][1]  # bottom line of peak_n
        end = locations_new[sec + 1][0]  # top line of peak_n+1
        mid_lines.append([beginning, end])
    first_line = locations_new[0][0] - int(thr_peaks[0]['lh'] / 20)  # first line found in the input image
    last_line = locations_new[-1][1] + int(thr_peaks[-1]['lh'] / 2)  # last line found in the input image

    plotImageAndHistLines(newImage, locations_new)
    return mid_lines, first_line, last_line, line_heights, hist, thr_num


def calc_outlier(data, method="std"):
    # method1: interquartile range
    if method == "iqr":
        q3, q1 = np.percentile(data, [75, 25])
        iqr = q3 - q1
        outlier = q1 - 1.5 * iqr

    # method2: standard deviation
    else:
        outlier = np.mean(data) - np.std(data)

    return outlier


def plotHist(hist, y_threshold):
    fs = 25
    plt.figure(figsize=(16, 12))
    plt.plot(hist)
    plt.axhline(y=y_threshold, color="r", linestyle="-")
    plt.ylim(0, max(hist) * 1.1)
    plt.xlabel("Row", fontsize=fs)
    plt.ylabel("Black pixels", fontsize=fs)
    plt.title("Binary image black pixel counting result", fontsize=fs)
    plt.yticks(fontsize=fs - 5)
    plt.xticks(fontsize=fs - 5)
    plt.grid()
    plt.show()


def plotImageAndHistLines(newImage, midlines, a_star_paths=None):
    plt.figure(figsize=(16, 12))
    plt.imshow(newImage, cmap="gray")
    if a_star_paths:
        for path in a_star_paths:
            plt.plot(path[:, 1], path[:, 0], color="g")
    for i in range(len(midlines)):
        for idx, loc in enumerate(midlines[i]):
            if idx == 0:
                plt.axhline(y=loc, color="r", linestyle="-")
            else:
                plt.axhline(y=loc, color="b", linestyle="-")
    plt.show()


def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary * 1
    return binary


def getMidSections(mid_lines, image):
    """ Returns a list of 2D arrays (pixels in a rectangle) that each correspond to a mid_line (break) """
    sections = []
    for sec in mid_lines:
        sections.append(image[int(sec[0]):int(sec[1]), :])
    return sections
# ------------------------- Histogram part: end-------------------------

# ------------------------- A* algorithm part ----------------------
def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def astar(array, start, goal):
    # 8 directions: up, down, right, left, ....
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # 5 directions (no backward movement)
    # neighbors = [(0,1),(0,-1),(1,0),(1,1),(1,-1)]

    close_set = set()
    came_from = {}  # prev step
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))

    return []


def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)


def path_exists(window_image):
    # very basic check first then proceed to A* check
    if 0 in horizontal_projections(window_image):
        return True

    padded_window = np.zeros((window_image.shape[0], 1))
    world_map = np.hstack((padded_window, np.hstack((window_image, padded_window))))
    path = np.array(
        astar(world_map, (int(world_map.shape[0] / 2), 0), (int(world_map.shape[0] / 2), world_map.shape[1])))
    if len(path) > 0:
        return True

    return False


def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False

    for col in range(nmap.shape[1]):
        start = col
        end = col + 20
        if end > nmap.shape[1] - 1:
            end = nmap.shape[1] - 1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
            road_blocks.append(col)
            print('roadblock found')

        if needtobreak == True:
            break

    return road_blocks


def group_the_road_blocks(road_blocks):
    # group the road blocks
    road_blocks_cluster_groups = []
    road_blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        road_blocks_cluster.append(value)
        if index < size - 1 and (road_blocks[index + 1] - road_blocks[index]) > 1:
            road_blocks_cluster_groups.append(
                [road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster) - 1]])
            road_blocks_cluster = []

        if index == size - 1 and len(road_blocks_cluster) > 0:
            road_blocks_cluster_groups.append(
                [road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster) - 1]])
            road_blocks_cluster = []

    return road_blocks_cluster_groups

# ------------------------- A* algorithm part: end ----------------------

image = rotateImage(img_path)
mid_lines, first_line, last_line, line_height, hist, thr_num = getLines(image)
# plotHist(hist, thr_num)
binary_image = get_binary(image)
mid_sections = getMidSections(mid_lines, binary_image)

# plt.figure(figsize=(20, 20))
# plt.imshow(mid_sections[6, :], cmap="gray")
# plt.show()

# count = 0
# for cluster_of_interest in mid_lines:
#     print(count)
#     count += 1
#     nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest) - 1], :]
#     road_blocks = get_road_block_regions(nmap)
#     road_blocks_cluster_groups = group_the_road_blocks(road_blocks)
#     # create the doorways
#     for index, road_blocks in enumerate(road_blocks_cluster_groups):
#         window_image = nmap[:, road_blocks[0]: road_blocks[1] + 10]
#         binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest) - 1], :][:,
#         road_blocks[0]: road_blocks[1] + 10][int(window_image.shape[0] / 2), :] *= 50
#     if count == 6:  # here im trying to see the effect of blasting through
#         plt.figure(figsize=(20, 20))
#         plt.imshow(binary_image[mid_lines[6][0]:mid_lines[6][len(mid_lines[6]) - 1], :], cmap="gray")
#         plt.show()
#
# line_segments = []
# for i, cluster_of_interest in tqdm(enumerate(mid_lines)):
#     nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest) - 1], :]
#     path = np.array(astar(nmap, (int(nmap.shape[0] / 2), 0), (int(nmap.shape[0] / 2), nmap.shape[1] - 1)))
#     offset_from_top = cluster_of_interest[0]
#     path[:, 0] += offset_from_top
#     line_segments.append(path)
#
# # -----------------------------------------------------------
#
# cluster_of_interest = mid_lines[1]
# offset_from_top = cluster_of_interest[0]
# nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest) - 1], :]
# plt.figure(figsize=(20, 20))
# plt.imshow(invert(nmap), cmap="gray")
#
# path = np.array(astar(nmap, (int(nmap.shape[0] / 2), 0), (int(nmap.shape[0] / 2), nmap.shape[1] - 1)))
# plt.plot(path[:, 1], path[:, 0])
#
# # -----------------------------------------------------------
#
# offset_from_top = cluster_of_interest[0]
# fig, ax = plt.subplots(figsize=(20, 10), ncols=2)
# for path in line_segments:
#     ax[1].plot((path[:, 1]), path[:, 0])
# ax[1].axis("off")
# ax[0].axis("off")
# ax[1].imshow(img, cmap="gray")
# ax[0].imshow(img, cmap="gray")
#
# # -----------------------------------------------------------
#
# ## add an extra line to the line segments array which represents the last bottom row on the image
# last_bottom_row = np.flip(
#     np.column_stack(((np.ones((img.shape[1],)) * img.shape[0]), np.arange(img.shape[1]))).astype(int), axis=0)
# line_segments.append(last_bottom_row)

########## I commented out the stuff we were using previously because i was able to give the program everything it needed i think


# for cluster_of_interest in hpp_clusters:
#     nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
#     road_blocks = get_road_block_regions(nmap)
#     road_blocks_cluster_groups = group_the_road_blocks(road_blocks)
#     #create the doorways
#     for index, road_blocks in enumerate(road_blocks_cluster_groups):
#         window_image = nmap[:, road_blocks[0]: road_blocks[1]+10]
#         binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:][:, road_blocks[0]: road_blocks[1]+10][int(window_image.shape[0]/2),:] *= 0

# # Segment all the lines using the A* algorithm

# line_segments = []
# for i, cluster_of_interest in enumerate(hpp_clusters):
#     nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
#     path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
#     offset_from_top = cluster_of_interest[0]
#     path[:,0] += offset_from_top
#     line_segments.append(path)


paths = []
line_count = 0
for section in tqdm(mid_sections):
    print('line number: ', line_count, 'section dimensions: ',section.shape)
    # nmap = image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
    path = np.array(astar(section, (int(section.shape[0] / 2), 0), (int(section.shape[0] / 2), section.shape[1] - 1)))
    if len(path.shape) == 2:
        # offset_from_top = int(section.shape[0] / 2) + mid_lines[line_count][0]
        offset_from_top = mid_lines[line_count][0]
        for idx in range(len(path[:, 0])):
            path[idx][0] += offset_from_top
        paths.append(path)
    line_count += 1

plotImageAndHistLines(binary_image, mid_lines, paths)
# plt.figure(figsize=(20, 20))
# plt.imshow(binary_image, cmap="gray")
# for path in paths:
#     plt.plot(path[:, 1], path[:, 0], color="g")
# plt.show()

# test