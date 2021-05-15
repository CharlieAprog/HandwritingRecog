from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.util import invert
from skimage.filters import threshold_otsu

import numpy as np
import PIL
from PIL import Image
import PIL.ImageOps
import imutils
import cv2
from matplotlib import cm
from matplotlib import pyplot as plt
import pprint
from heapq import *

# img_path = 'handwritten1.jpg'
img_path = 'data/image-data/binary/P423-1-Fg002-R-C02-R01-binarized.jpg'

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
    thr_num = max(hist) * 0.2
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

    mid_lines = []
    for sec in range(len(thr_peaks) - 1):
        beginning = thr_peaks[sec]['loc'][1]  # bottom line of peak_n
        end = thr_peaks[sec + 1]['loc'][0]  # top line of peak_n+1
        mid_lines.append([beginning, end])
    top_line = thr_peaks[sec]['loc'][0] - thr_peaks[sec]['lh'] / 2
    bottom_line = thr_peaks[sec]['loc'][-1] + thr_peaks[sec]['lh'] / 2
    return mid_lines, top_line, bottom_line, line_heights

def plotHist(hist):
    fs = 25
    plt.figure(figsize=(16, 12))
    plt.plot(hist)
    plt.ylim(0, max(hist) * 1.1)
    plt.xlabel("Row", fontsize=fs)
    plt.ylabel("Black pixels", fontsize=fs)
    plt.title("Binary image black pixel counting result", fontsize=fs)
    plt.yticks(fontsize=fs - 5)
    plt.xticks(fontsize=fs - 5)
    plt.grid()
    plt.show()


def plotImageAndHistLines(newImage, thr_peaks, line_heights):
    plt.figure(figsize=(16, 12))
    plt.imshow(newImage)
    avg_lh = sum(line_heights) / len(line_heights)
    q3, q1 = np.percentile(line_heights, [75, 25])
    iqr = q3 - q1
    outlier = q1 - 1.5 * iqr
    for i in range(len(thr_peaks)):
        if not thr_peaks[i]["lh"] <= outlier:
            for idx, loc in enumerate(thr_peaks[i]["loc"]):
                if idx == 0:
                    plt.axhline(y=loc - avg_lh // 3, color="r", linestyle="-")
                else:
                    plt.axhline(y=loc + avg_lh // 2.5, color="r", linestyle="-")
    plt.show()


# def plotImageAndLines(image, lines):
#     plt.figure(figsize=(16, 12))
#     plt.imshow(image)
#     for line in lines:
#         for p in line:
#             plt.plot(p)
#     plt.show()

def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary * 1
    return binary

def getMidSections(mid_lines, image):
    """ Mid section between 2 lines that contain characters """
    sections = []
    for sec in mid_lines:
        sections.append(image[sec[0]:sec[1], :])
    return sections
# ------------------------- Histogram part -------------------------

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


# ------------------------- A* algorithm part ----------------------

image = rotateImage(img_path)
mid_lines, top_line, bottom_line, line_height = getLines(image)
binary_image = get_binary(image)
mid_sections = getMidSections(mid_lines, binary_image)

# Segment all the lines using the A* algorithm
line_segments = []
# for section in mid_sections:
# nmap = image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
section = mid_sections[3]
offset_from_top = section[0]

plt.figure(figsize=(20, 20))
plt.imshow(invert(section), cmap="gray")
path = np.array(astar(section, (int(section.shape[0] / 2), 0), (int(section.shape[0] / 2), section.shape[1] - 1)))
plt.plot(path[:, 1], path[:, 0])
plt.show()
offset_from_top = section[0]
# path[:, 0] += offset_from_top
line_segments.append(path)


# line_segments = []
# # len(hpp_clusters): number of extracted segments (lines)
# for section in mid_sections:
#     offset_from_top = section[0]
#     path = np.array(astar(section, (int(section.shape[0] / 2), 0), (int(section.shape[0] / 2), section.shape[1] - 1)))
#     path[:, 0] += offset_from_top
#     line_segments.append(path)

