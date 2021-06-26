import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.filters import threshold_otsu
import imutils
import os
import csv
import glob
import copy
from findpeaks import findpeaks
from Text_Segmentation.plotting import *


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


def get_image(img_path, hough_transform=True):
    """ Reads and returns a binary image from given path. """
    image = cv2.imread(img_path, 0)
    if hough_transform:
        image = cv2.bitwise_not(image)
    assert len(image.shape) == 2, "Trying to read image while being in a wrong folder, or provided path is wrong."
    plt.imshow(image, cmap="gray")
    plt.title("Image right after reading")
    plt.show()
    return image


def get_binary(img):
    """ Returns a binarized image being undergone 'threshold_otsu()' """
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:  # Fully black or white image
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary * 1
    return binary.astype(np.uint8)


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
    h_hist = [len(row.nonzero()[0]) for row in new_image]

    fp = findpeaks(lookahead=3)
    result = fp.fit(h_hist)
    peaks = result['df'][result['df']['peak'] == True]
    val_peaks = peaks.y.to_list()
    fp.plot()

    # Identify histogram peaks
    thr_lines = {}  # thresholded lines; containing lines of interest and the horizontal projections thereof
    local_lines = []  # list of pixels in a peak's neighborhood from left to right
    c = 0  # counter variable
    thr_num = np.mean(val_peaks)
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

    # B) Obtain the starting and ending locations, and the height of each peak's neighborhood (section)
    locations = []
    for idx, line in enumerate(thr_lines.items()):
        y_loc_start = line[0]
        line_projections = line[1]  # line projections within the neighbourhood of the current peak
        height = len(line_projections)
        locations.append([y_loc_start, y_loc_start + height])  # starting and ending location of the section

    section_heights = [x[1] - x[0] for x in locations]
    avg_sh = np.mean(section_heights)
    buffer = int(avg_sh * 0.5)
    fp = findpeaks(lookahead=20)
    result = fp.fit(h_hist)
    peaks = result['df'][result['df']['peak'] == True]
    loc_peaks = peaks.x.to_list()
    locations_extended = copy.deepcopy(locations)
    for peak in loc_peaks:
        contained = False
        for loc in locations_extended:
            if loc[1] > peak > loc[0]:
                contained = True
        if not contained:
            locations_extended.append([peak - buffer, peak + buffer])
    locations_extended = sorted(locations_extended, key=lambda x: x[0])

    # C) Combining sections that are too close together
    # distances between consecutive sections (SEC_n+1_top - SEC_n_bottom)
    distances = [
        locations_extended[sec + 1][0] - locations_extended[sec][1]
        for sec in range(len(locations_extended) - 1)
    ]
    min_distance = calc_outlier(distances) if calc_outlier(distances) > 18 else 18
    # min_distance = 18
    # # print(distances, '\n', min_distance)
    #
    locations_extended_new = []
    idx = 0
    while idx < len(locations_extended):  # Run combination algorithm from top to bottom
        if idx < len(distances):  # len(distances) = len(locations_extended)-1
            distance = distances[idx]
            locations_extended_new.append(locations_extended[idx])
        else:  # End of algorithm
            if locations_extended[idx][1] - locations_extended[idx][0] > min_distance:
                locations_extended_new.append(locations_extended[idx])
            break

        idx2 = 1
        while distance < min_distance:
            # don't manipulate lines that have been added after post-detection
            if locations_extended[idx] not in locations:
                idx2 += 1
                break
            locations_extended_new[-1][1] = locations_extended[idx + idx2][1]  # merge current location with next
            if idx + idx2 < len(distances):
                distance = distances[idx + idx2]  # merge current distance with next
            else:
                break
            idx2 += 1
        idx += idx2

    # D) Adding buffer, based on average height of the NEW (!) sections, to each section that is too small
    section_heights_new = [x[1] - x[0] for x in locations_extended_new]
    avg_sh_new = np.mean(section_heights_new)
    for idx, loc in enumerate(locations_extended_new):
        if section_heights_new[idx] < avg_sh_new:
            for i in range(len(loc)):
                if idx == 0:
                    # top lines are pushed up
                    locations_extended_new[idx][i] -= int(avg_sh_new / 5)
                else:
                    # bottom lines are pushed down
                    locations_extended_new[idx][i] += int(avg_sh_new / 6)

    # E) Remove sections with too small height (may occur due to very short lines with scattered artefacts)
    locations_final = []
    min_height = int(calc_outlier(section_heights_new) / 3) + 2
    for idx in range(len(section_heights_new)):
        if section_heights_new[idx] >= min_height:
            locations_final.append(locations_extended_new[idx])
    section_heights_final = [x[1] - x[0] for x in locations_final]
    avg_sh_final = np.mean(section_heights_final)

    # F) Obtaining the locations of the inbetween (non-character) sections
    # and adding a bonus line to the top of the image
    mid_lines = []
    top_line = [
        locations_final[0][0] - int(avg_sh_final * 2.5),
        locations_final[0][0] - int(avg_sh_final)
    ]
    if top_line[0] <= 0 or top_line[1] <= 0:
        top_line[0] = 0
        top_line[1] = 1
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
    if bottom_line[0] >= new_image.shape[0] or bottom_line[1] >= new_image.shape[0]:
        bottom_line[0] = new_image.shape[0]-1
        bottom_line[1] = new_image.shape[0]
    mid2.append(bottom_line)
    return mid2, avg_sh_final, h_hist, thr_num

    # return locations, h_hist


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
    image = get_binary(rotate_image(get_image(img_path, hough_transform=True)))
    dilated_image = copy.deepcopy(image)
    kernel = np.ones((2, 2), 'uint8')
    dilated_image = cv2.dilate(dilated_image, kernel, iterations=1)
    plot_simple_images([image, dilated_image], title="Original vs Dilated image")
    if not os.path.exists(new_folder_path):
        print("Running line segmentation on new image...")
        os.makedirs(new_folder_path)
        # run image-processing
        # mid_lines, hist = get_lines(dilated_image)
        mid_lines, avg_lh, hist, thr_num = get_lines(dilated_image)
        plot_hist(hist,
                  thr_num,
                  save=True,
                  folder_path=new_folder_path,
                  overwrite_path=False)
        plot_hist_lines_on_image(dilated_image,
                                 mid_lines,
                                 save=True,
                                 folder_path=new_folder_path,
                                 overwrite_path=False)
        # Find paths with A*
        paths = find_paths(mid_lines, dilated_image, avg_lh)
        plot_paths_next_to_image(dilated_image,
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
        assert len(paths) > 0, "Trying to load paths from an empty folder, delete folder and run A* again."

    section_images = []
    line_count = len(paths)
    for line_index in range(line_count -
                            1):  # |-------- extract sections from loaded paths
        section_image = extract_char_section_from_image(dilated_image, paths[line_index],
                                                        paths[line_index + 1])
        section_images.append(section_image)
    return section_images


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

    # plot_hough_transform(hspace, theta, dist, x0, x1, origin,image,new_image)
    # Compute difference between the two lines
    angle_difference = np.max(angles) - np.min(angles)

    return new_image


# |-------------------------|
# |          A*             |
# |-------------------------|
import numpy as np
from heapq import *
from tqdm import tqdm


def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def astar(array, start, goal, i):
    # 8 directions: up, down, right, left, ....
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1),
                 (-1, -1)]
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
                else:  # array bound y walls
                    continue
            else:  # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(
                    neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [
                i[1] for i in oheap
            ]:
                if i == 11:
                    print(current)
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + \
                                   heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
    return []


def horizontal_projections(sobel_image):
    return np.sum(sobel_image, axis=1)


def path_exists(window_image):
    # very basic check first then proceed to A* check
    if 0 in horizontal_projections(window_image):
        return True

    padded_window = np.zeros((window_image.shape[0], 1))
    world_map = np.hstack(
        (padded_window, np.hstack((window_image, padded_window))))
    path = np.array(
        astar(world_map, (int(world_map.shape[0] / 2), 0),
              (int(world_map.shape[0] / 2), world_map.shape[1]), 0))
    if len(path) > 0:
        return True

    return False


def get_road_block_regions(nmap):
    road_blocks = []
    needtobreak = False

    for col in range(nmap.shape[1]):
        start = col
        end = col + 40
        if end > nmap.shape[1] - 1:
            end = nmap.shape[1] - 1
            needtobreak = True

        if path_exists(nmap[:, start:end]) == False:
            road_blocks.append(col)

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
        # split up the clusters
        if index < size - 1 and (road_blocks[index + 1] - road_blocks[index]) > 1 or \
                index == size - 1 and len(road_blocks_cluster) > 0:
            road_blocks_cluster_groups.append(
                [road_blocks_cluster[0], road_blocks_cluster[-1]])
            road_blocks_cluster = []
    return road_blocks_cluster_groups


def find_paths(hpp_clusters, binary_image, avg_lh):
    fake_rb_indices = []
    agent_height = []
    upward_push = int(avg_lh * 0.85)
    for idx, cluster_of_interest in enumerate(hpp_clusters):
        print(idx)
        if idx == 2:
            asd = 0
        nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[-1]]
        road_blocks = get_road_block_regions(nmap)
        start_end_height = int(nmap.shape[0] / 2)
        agent_height.append(start_end_height)

        # check for fake roadblocks
        if len(road_blocks) != 0:
            nmap_rb = binary_image[cluster_of_interest[0] - upward_push:cluster_of_interest[-1]]
            road_blocks_new = get_road_block_regions(nmap_rb)
            if road_blocks_new != road_blocks and len(road_blocks_new) < len(road_blocks):
                print('Fake roadblock has been hit, better path found')
                fake_rb_indices.append(idx)
                road_blocks = road_blocks_new
        road_blocks_cluster_groups = group_the_road_blocks(road_blocks)

        # create the doorways for real roadblocks
        for road_blocks in road_blocks_cluster_groups:
            rb_end_reached = False  # true end of the roadblock
            i = 0
            mid = (cluster_of_interest[1]-cluster_of_interest[0])//2
            prev_pixel = binary_image[
                         cluster_of_interest[0]:
                         cluster_of_interest[-1], :][:, road_blocks[0]:binary_image.
                                                                           shape[1] - 1][mid, 0]
            # making sure prev_pixel is initiated with a 0
            step_back = 1
            while prev_pixel:
                prev_pixel = binary_image[
                             cluster_of_interest[0]:
                             cluster_of_interest[-1], :][:, road_blocks[0] -
                                                            step_back:binary_image.
                                                                          shape[1] - 1][mid, 0]
                step_back += 1
            assert prev_pixel == 0, "prev_pixel=1 at the start of annulling, horizontal cut cannot be performed"

            while True:
                if np.sum(binary_image[cluster_of_interest[0]:cluster_of_interest[-1], :][:,
                    road_blocks[0]: binary_image.shape[1] - 1][mid, :]) == 0:
                    print("Cut omitted, path is free.")
                    break
                i += 1
                # If roadblock end found (black pixel)
                if binary_image[cluster_of_interest[0]:cluster_of_interest[-1], :][:,
                    road_blocks[0]: binary_image.shape[1] - 1][mid, i] == 0:
                    if prev_pixel == 1:
                        rb_end_reached = True
                        binary_image[
                        cluster_of_interest[0]:cluster_of_interest[
                            -1], :][:,
                        road_blocks[0]:binary_image.shape[1] -
                                       1][mid, 0:i] = 0
                    if rb_end_reached:
                        # detect fake roadblock end
                        fake_end_length = 40
                        if len(
                                np.nonzero(
                                    binary_image[cluster_of_interest[0]:
                                    cluster_of_interest[-1], :]
                                    [:, road_blocks[0]:binary_image.shape[1]][
                                    mid, i:i + fake_end_length])[0]) != 0:
                            rb_end_reached = False
                            prev_pixel = 0
                            print("fake end")
                            continue
                        # true end
                        print(f"Cuts have been made for line {idx}, roadblock {road_blocks}")
                        break
                    prev_pixel = 0
                else:
                    prev_pixel = 1

            # Plot enlargened section (if needed) where horizontal cut is performed
            # if idx == 6:
            #     plt.plot(figsize=(16, 12))
            #     plt.imshow(invert(binary_image[cluster_of_interest[0]:cluster_of_interest[-1],:][:, road_blocks[0]: road_blocks[1]]), cmap="gray")
            #     plt.show()

    paths = []
    for i, cluster_of_interest in tqdm(enumerate(hpp_clusters)):
        if i in fake_rb_indices:
            nmap = binary_image[cluster_of_interest[0] -
                                upward_push:cluster_of_interest[-1]]
            offset_from_top = cluster_of_interest[0] - upward_push
            height = agent_height[i] + upward_push
        else:
            nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[-1]]
            offset_from_top = cluster_of_interest[0]
            height = agent_height[i]
        path = np.array(
            astar(nmap, (height, 0), (height, nmap.shape[1] - 1), i))
        print("path.shape:", path.shape)
        # assert path.shape[0] != 0, "Path has shape (0,), algorithm failed to reach destination."
        if path.shape[0] == 0:
            print(f'Path could not be generated for line {i}')
            continue
        path[:, 0] += offset_from_top
        path = [list(step) for step in path]
        paths.append(path)
    return paths


# |-------------------------|
# |          A*             |
# |-------------------------|

# image_names = ["25-Fg001.pbm", "124-Fg004.pbm", "archaic1.jpg", "archaic2.jpg", "archaic3.jpg",
#                 "hasmonean3.jpg", "hasmonian1.jpg", "herodian1.jpg", "herodian2.jpg", "herodian3.jpg"]
# for image_name in image_names:
#     # image_name = "25-Fg001.pbm"
#     dev_path = f"../data/cropped_labeled_images/{image_name}"  # development path
#     new_folder_path = f"../data/cropped_labeled_images/paths/{image_name[0:-4]}"
#     section_images = line_segmentation(dev_path, new_folder_path)

for i in [12, 16]:
    image_name = i
    dev_path = f"../data/image-data/binaryRenamed/{image_name}.jpg"  # development path
    new_folder_path = f"../data/image-data/binaryRenamed/paths/{str(image_name)}"
    section_images = line_segmentation(dev_path, new_folder_path)

# def rotate_image(image):
#     # tested_angles = np.linspace(np.pi* 49/100, np.pi *51/100, 100)
#     tested_angles = np.linspace(-np.pi * 45 / 100, -np.pi * 55 / 100, 100)
#     hspace, theta, dist, = hough_line(image, tested_angles)
#
#     h, q, d = hough_line_peaks(hspace, theta, dist)
#
#     #################################################################
#     # Example code from skimage documentation to plot the detected lines
#     angle_list = []  # Create an empty list to capture all angles
#     dist_list = []
#     # Generating figure 1
#     fig, axes = plt.subplots(1, 4, figsize=(15, 6))
#     ax = axes.ravel()
#     ax[0].imshow(image, cmap='gray')
#     ax[0].set_title('Input image')
#     ax[0].set_axis_off()
#     ax[1].imshow(np.log(1 + hspace),
#                  extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
#                  cmap='gray', aspect=1 / 1.5)
#     ax[1].set_title('Hough transform')
#     ax[1].set_xlabel('Angles (degrees)')
#     ax[1].set_ylabel('Distance (pixels)')
#     ax[1].axis('image')
#     ax[2].imshow(image, cmap='gray')
#     origin = np.array((0, image.shape[1]))
#
#     for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist, min_distance=50, threshold=0.76 * np.max(hspace))):
#         angle_list.append(angle)  # Not for plotting but later calculation of angles
#         dist_list.append(dist)
#         y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
#         ax[2].plot(origin, (y0, y1), '-r')
#
#     ave_angle = np.mean(angle_list)
#     ave = ave_angle * 180 / np.pi
#     ave_dist = np.mean(dist_list)
#     x0, x1 = (ave_dist - origin * np.cos(ave_angle)) / np.sin(ave_angle)
#
#     ax[2].plot(origin, (x0, x1), '-b')
#     ax[2].set_xlim(origin)
#     ax[2].set_ylim((image.shape[0], 0))
#     ax[2].set_axis_off()
#     ax[2].set_title('Detected lines')
#
#     ###############################################################
#     # Convert angles from radians to degrees (1 rad = 180/pi degrees)
#     angles = [a * 180 / np.pi for a in angle_list]
#     change = 90 - -1 * np.mean(angles)
#     newImage = imutils.rotate_bound(image, -change)
#     ax[3].imshow(newImage, cmap='gray')
#     plt.tight_layout()
#     plt.show()
#
#     # plotHoughTransform(hspace, theta, dist, x0, x1, origin, newImage)
#
#     # Compute difference between the two lines
#     angle_difference = np.max(angles) - np.min(angles)
#
#     return newImage
