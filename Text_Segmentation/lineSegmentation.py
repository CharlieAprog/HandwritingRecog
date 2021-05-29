import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.filters import threshold_otsu
import imutils


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


def getImage(img_path):
    image = cv2.imread(img_path, 0)
    image = cv2.bitwise_not(image)
    return image


def get_binary(img):
    mean = np.mean(img)
    if mean == 0.0 or mean == 1.0:
        return img

    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = binary * 1
    return binary


# ------------------------- Hough Transform -------------------------


def rotateImage(image):
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
    newImage = cv2.bitwise_not(imutils.rotate_bound(image, -change))

    # plotHoughTransform(hspace, theta, dist, x0, x1, origin, newImage)

    # Compute difference between the two lines
    angle_difference = np.max(angles) - np.min(angles)

    return newImage


# ------------------------- Hough Transform -------------------------

# ------------------------- Histogram part -------------------------


@timer
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
        elif len(
                temp
        ) > 0:  # once p is nolonger larger than thresh and it has been in the past
            # add temp to a dict and start temp again
            thr.setdefault(col - c, temp)
            temp = []
            c = 0

    line_heights = []
    thr_peaks = []
    for idx, p in enumerate(thr.items()):
        line_heights.append(len(p[1]))
        thr_peaks.append({
            "loc": [p[0], p[0] + len(p[1])],
            "value": max(p[1]),
            "lh": p[0] + len(p[1]) - p[0]
        })

    # combining lines that are too close together
    locations = [x['loc'] for x in thr_peaks]
    distances = [
        locations[sec + 1][0] - locations[sec][1]
        for sec in range(len(locations) - 1)
    ]
    min_distance = calc_outlier(
        distances) if calc_outlier(distances) > 18 else 18
    print(distances, '\n', min_distance)
    # min_distance = int(max(distances) /6) if int(max(distances) /6) > 22 else 22
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
            locations_new[-1][second] = locations[idx + idx2][second]
            if idx + idx2 < len(distances):
                distance = distances[idx + idx2]
            else:
                break
            idx2 += 1
        idx += idx2

    # Adding buffer, based on average height of the NEW (!) lines, to each line that is too small
    line_heights_new = [x[1] - x[0] for x in locations_new]
    avg_lh_new = np.mean(line_heights_new)
    for idx, loc in enumerate(locations_new):
        if line_heights_new[idx] < avg_lh_new:
            for i in range(len(loc)):
                if idx == 0:
                    # top lines are pushed up
                    locations_new[idx][i] -= int(avg_lh_new / 5)
                else:
                    # bottom lines are pushed down
                    locations_new[idx][i] += int(avg_lh_new / 6)

    # remove sections that are left over than have a low hight
    locations_final = []
    mid_distances = [line[1] - line[0] for line in locations_new]
    min_height = int(calc_outlier(mid_distances) / 3) + 2
    for idx in range(len(mid_distances)):
        if mid_distances[idx] >= min_height:
            locations_final.append(locations_new[idx])
    line_heights_final = [x[1] - x[0] for x in locations_final]
    avg_lh_final = np.mean(line_heights_final)

    # obtaining the locations of the inbetween sections
    mid_lines = []
    top_line = [
        locations_final[0][0] - int(avg_lh_final * 2.5),
        locations_final[0][0] - int(avg_lh_final)
    ]
    mid_lines.append(top_line)
    for sec in range(len(locations_final) - 1):
        if locations_final[sec][first] < locations_final[sec][second]:
            beginning = locations_final[sec][1]  # bottom line of peak_n
            end = locations_final[sec + 1][0]  # top line of peak_n+1
            mid_lines.append([beginning, end])

    # Obtain bottom line
    mid2 = []
    for sec in mid_lines:
        if sec[0] < sec[1]:
            mid2.append(sec)
    bottom_line = [
        locations_final[-1][1] + int(avg_lh_final),
        locations_final[-1][1] + int(avg_lh_final * 2)
    ]
    mid2.append(bottom_line)

    return mid2, top_line, bottom_line, avg_lh_final, hist, thr_num,


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
