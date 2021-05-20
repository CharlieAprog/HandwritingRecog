from numpy.core.fromnumeric import nonzero
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
from skimage.util import invert
from skimage.filters import threshold_otsu
from skimage.filters import sobel


import numpy as np
import csv
import os
import glob
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


# ------------------------- Plotting functions -------------------------
def plotHist(hist, y_threshold):
    fs = 25
    plt.figure(figsize=(10, 6))
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


def plotHistLinesOnImage(newImage, midlines):
    plt.figure(figsize=(10, 6))
    plt.imshow(newImage, cmap="gray")
    for i in range(len(midlines)):
        for idx, loc in enumerate(midlines[i]):
            if idx == 0:
                plt.axhline(y=loc, color="r", linestyle="-")
            else:
                plt.axhline(y=loc, color="b", linestyle="-")
    plt.show()


def plotPathsNextToImage(binary_image, line_segments):
    fig, ax = plt.subplots(figsize=(10,6), ncols=2)
    for path in line_segments:
        ax[1].plot((path[:,1]), path[:,0])
    ax[1].axis("off")
    ax[0].axis("off")
    ax[1].imshow(binary_image, cmap="gray")
    ax[0].imshow(binary_image, cmap="gray")
    plt.show()
# ------------------------- Plotting functions -------------------------

# ------------------------- Hough Transform -------------------------
def rotateImage(img_path):
    image = cv2.imread(img_path, 0)
    image = cv2.bitwise_not(image)
    # tested_angles = np.linspace(np.pi* 49/100, np.pi *51/100, 100)
    tested_angles = np.linspace(-np.pi * 40 / 100, -np.pi * 50 / 100, 100)
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

    for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist, min_distance=50, threshold=0.76 * np.max(hspace))):
        angle_list.append(angle)  # Not for plotting but later calculation of angles
        dist_list.append(dist)
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        # ax[2].plot(origin, (y0, y1), '-r')

    ave_angle = np.mean(angle_list)
    ave = ave_angle * 180 / np.pi
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

    return newImage
# ------------------------- Hough Transform -------------------------

img_path = 'data/image-data/binaryRenamed/1.jpg'

# ------------------------- Histogram part -------------------------
def getLines(newImage):
    hist = []
    row_len = newImage.shape[1]
    for row in newImage:
        hist.append(row_len - len(row.nonzero()[0]))

    temp = []  # list of pixels in a peak's neighborhood from left to right
    thr = {}  # dictionary containing lines of interest
    c = 0  # counter variable
    thr_num = max(hist) * 0.09
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

    # Filtering line height outliers
    # min_height = calc_outlier(line_heights)
    # thr_peaks_filtered = []
    # for i in range(len(thr_peaks)):
    #     if thr_peaks[i]["lh"] > min_height:
    #         thr_peaks_filtered.append(thr_peaks[i])

    #combining lines that are too close together
    locations = [x['loc'] for x in thr_peaks]
    distances = [locations[sec + 1][0] - locations[sec][1] for sec in range(len(locations) - 1)]
    min_distance = calc_outlier(distances) if calc_outlier(distances)  > 18 else 18
    print(distances,'\n', min_distance)
    #min_distance = int(max(distances) /6) if int(max(distances) /6) > 22 else 22
    locations_new = []
    idx = 0
    first = 0
    second = 1
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
            locations_new[-1][second] = locations[idx+idx2][second]
            if idx + idx2 < len(distances):
                distance = distances[idx + idx2]
            else:
                break
            idx2 += 1
        idx += idx2

    # Adding buffer, based on average height of the NEW (!) lines, to each line that is too small
    line_heights_new = [x[1]-x[0] for x in locations_new]
    avg_lh = np.mean(line_heights_new)
    for idx, loc in enumerate(locations_new):
        if line_heights_new[idx] < avg_lh:
            for i in range(len(loc)):
                if idx == 0:
                    locations_new[idx][i] -= int(avg_lh / 5 ) # top lines are pushed up
                else:
                    locations_new[idx][i] += int(avg_lh / 6)  # bottom lines are pushed down

    #remove sections that are left over than have a low hight
    locations_final = []
    mid_distances = [line[1] - line[0] for line in locations_new]
    min_height = int(calc_outlier(mid_distances) /3) + 2
    for idx in range(len(mid_distances)):
        if mid_distances[idx] >= min_height:
            locations_final.append(locations_new[idx])

    #obtaining the locations of the inbetween sections
    mid_lines = []
    top_line = [locations_final[0][0] - int(avg_lh*2.5) ,locations_final[0][0] - int(avg_lh)]
    mid_lines.append(top_line)
    for sec in range(len(locations_final) - 1):
        if locations_final[sec][first] < locations_final[sec][second]:
                beginning = locations_final[sec][1]  # bottom line of peak_n
                end = locations_final[sec + 1][0]  # top line of peak_n+1
                mid_lines.append([beginning, end])

    mid2 = []
    for sec in mid_lines:
        if sec[0] < sec[1]:
            mid2.append(sec)

    
    bottom_line = [locations_final[-1][1] + int(avg_lh), locations_final[-1][1]+ int(avg_lh*2)]
    mid2.append(bottom_line)

    

    return mid2, top_line, bottom_line, line_heights, hist, thr_num


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


def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary * 1
    return binary

# def getMidSections(mid_lines, image):
#     """ Mid section between 2 lines that contain characters """
#     sections = []
#     for sec in mid_lines:
#         sections.append(image[sec[0]:sec[1], :])
#     return sections
# ------------------------- Histogram part -------------------------

# ------------------------- A* algorithm part ----------------------
def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def astar(array, start, goal,i):
    # 8 directions: up, down, right, left, ....
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
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
            if i == 11:
                    print(current)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else: # array bound y walls
                    continue
            else: # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                if i == 11:
                    print(current)
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
    return []


def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1) 


def path_exists(window_image):
    #very basic check first then proceed to A* check
    if 0 in horizontal_projections(window_image):
        return True
    
    padded_window = np.zeros((window_image.shape[0],1))
    world_map = np.hstack((padded_window, np.hstack((window_image,padded_window)) ) )
    path = np.array(astar(world_map, (int(world_map.shape[0]/2), 0), (int(world_map.shape[0]/2), world_map.shape[1]),0))
    if len(path) > 0:
        return True
    
    return False


def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False
    
    for col in range(nmap.shape[1]):
        start = col
        end = col+80
        if end > nmap.shape[1]-1:
            end = nmap.shape[1]-1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
            road_blocks.append(col)

        if needtobreak == True:
            break
            
    return road_blocks


def group_the_road_blocks(road_blocks):
    #group the road blocks
    road_blocks_cluster_groups = []
    road_blocks_cluster = []
    size = len(road_blocks)
    for index, value in enumerate(road_blocks):
        road_blocks_cluster.append(value)
        if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1: #split up the clusters 
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

        if index == size-1 and len(road_blocks_cluster) > 0: 
            road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
            road_blocks_cluster = []

    return road_blocks_cluster_groups

def find_paths(hpp_clusters, binary_image):
    count = 0
    for cluster_of_interest in hpp_clusters:
        print(count)
        count += 1
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
        road_blocks = get_road_block_regions(nmap)
        road_blocks_cluster_groups = group_the_road_blocks(road_blocks)
        #create the doorways
        for index, road_blocks in enumerate(road_blocks_cluster_groups):
            window_image = nmap[:, road_blocks[0]: road_blocks[1]+40]
            binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:][:, road_blocks[0]: road_blocks[1]+40 - 1][int(window_image.shape[0]/4),:] *= 0
            # if count == 6 or count == 6:
            #     fig, axes = plt.subplots(1, 4, figsize=(10, 6))
            #     ax = axes.ravel()
            #     ax[1].imshow(invert(binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:][:, road_blocks[0]: road_blocks[1]+40]), cmap="gray")
            #     plt.show()
    

    paths = []
    for i, cluster_of_interest in tqdm(enumerate(hpp_clusters)):
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
        path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1),i))
        offset_from_top = cluster_of_interest[0]
        print(path.shape)
        if path.shape[0] == 0:
            print('jeez')
            continue 
        path[:,0] += offset_from_top
        path = [list(step) for step in path]
        paths.append(path)

    # TODO: Replace this with the top/bottom line that we already have from getLine()
    # add an extra line to the line segments array which represents the last bottom row on the image
    # last_bottom_row = np.flip(np.column_stack(
        # ((np.ones((binary_image.shape[1],)) * binary_image.shape[0]), np.arange(binary_image.shape[1]))).astype(int),
                            #   axis=0)
    # paths.append(last_bottom_row)
    return paths


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

# ------------------------- Obtain paths ----------------------

image_num = 10
img_path = f'data/image-data/binaryRenamed/{image_num}.jpg'
new_folder_path = f"data/image-data/paths/{os.path.basename(img_path).split('.')[0]}"
image = rotateImage(img_path)
binary_image = get_binary(image)


if not os.path.exists(new_folder_path):
    # run image-processing
    mid_lines, top_line, bottom_line, line_height, hist, thr_num = getLines(image)
    plotHist(hist, thr_num)
    plotHistLinesOnImage(binary_image, mid_lines)
    paths = find_paths(mid_lines, binary_image)

    # # save paths
    os.makedirs(new_folder_path)
    for idx, path in enumerate(paths):
        save_path(path, f"{new_folder_path}/path_{idx}.csv")
else:
    # load paths
    file_paths_list = sorted(glob.glob(f'{new_folder_path}/*.csv'), key=get_key)
    paths = [] # a* paths
    sections_loaded = []
    for file_path in file_paths_list:
        line_path = load_path(file_path)
        paths.append(line_path)
        #binary_image[]
        

# extract sections from binary image determined by path
line_images = []
def extract_line_from_image(image, lower_line, upper_line):
    lower_boundary = np.min(lower_line[:, 0])
    upper_boundary = np.min(upper_line[:, 0])
    img_copy = np.copy(image)
    r, c = img_copy.shape
    for index in range(c-1):
        img_copy[0:lower_line[index, 0], index] = 0
        img_copy[upper_line[index, 0]:r, index] = 0
    
    return img_copy[lower_boundary:upper_boundary, :]

#plot sections
line_count = len(paths)
# fig, ax = plt.subplots(figsize=(10,10), nrows=line_count-1)
for line_index in range(line_count-1):
    line_image = extract_line_from_image(binary_image, paths[line_index], paths[line_index+1])
    line_images.append(line_image)
#     ax[line_index].imshow(invert(line_image), cmap="gray")

# plt.show()


from skimage.filters import threshold_otsu

#binarize the image, guassian blur will remove any noise in the image
first_line = line_images[1]
thresh = threshold_otsu(first_line)
binary = first_line > thresh

# find the vertical projection by adding up the values of all pixels along rows
vertical_projection = np.sum(binary, axis=0)

# plot the vertical projects
fig, ax = plt.subplots(nrows=2, figsize=(20,10))
plt.xlim(0, first_line.shape[1])
ax[0].imshow(binary, cmap="gray")
ax[1].plot(vertical_projection)
plt.show()

height = first_line.shape[0]

## we will go through the vertical projections and 
## find the sequence of consecutive white spaces in the image
whitespace_lengths = []
whitespace = 0

for idx in range(len(vertical_projection)):
    if vertical_projection[idx] == 0:
        whitespace = whitespace + 1
    elif vertical_projection[idx] != 0:
        if whitespace != 0:
            whitespace_lengths.append(whitespace) 
        whitespace = 0 # reset whitepsace counter. 
    if idx == len(vertical_projection)-1:
        print('yee')
        whitespace_lengths.append(whitespace)


print("whitespaces:", whitespace_lengths)
avg_white_space_length = np.mean(whitespace_lengths)
print("average whitespace lenght:", avg_white_space_length)

num_of_spaces = [x for x in whitespace_lengths if x > avg_white_space_length]
print(num_of_spaces)
## find index of whitespaces which are actually long spaces using the avg_white_space_length
whitespace_length = 0
divider_indexes = []

for index, vp in enumerate(vertical_projection):
    if vp == 0:
        whitespace_length = whitespace_length + 1
        if len(divider_indexes) == len(num_of_spaces) - 1  and whitespace_length > avg_white_space_length:
            divider_indexes.append(index+int(whitespace_length/2))
    elif vp != 0:
        if whitespace_length != 0 and whitespace_length > avg_white_space_length:
            divider_indexes.append(index-int(whitespace_length/2))
            whitespace_length = 0 # reset it
    
print(divider_indexes)

# lets create the block of words from divider_indexes
divider_indexes = np.array(divider_indexes)
dividers = np.column_stack((divider_indexes[:-1],divider_indexes[1:]))

# now plot the findings
fig, ax = plt.subplots(nrows=len(dividers), figsize=(5,6))
for index, window in enumerate(dividers):
    ax[index].axis("off")
    ax[index].imshow(first_line[:,window[0]:window[1]], cmap="gray")
   
plt.show()
#plotPathsNextToImage(binary_image, paths)

#plotPathsNextToImage(binary_image, paths)
# - get min y of upper path  and max Y of lower path coordinates
# - create a numpy array with max-min height and width = path_width
#     - loop through the rows of the new array
#         - fill in the new array with zeros everywhere, except at the indices where the loaded section has values

