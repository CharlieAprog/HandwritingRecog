from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
import numpy as np
import PIL
from PIL import Image
import PIL.ImageOps 
import imutils
import cv2
from matplotlib import cm
from matplotlib import pyplot as plt

img_path = 'data/image-data/binary/P423-1-Fg002-R-C02-R01-binarized.jpg'
#image = Image.open(img_path)
#image = PIL.ImageOps.invert(image)
image = cv2.imread(img_path, 0)
image = cv2.bitwise_not(image)

# tested_angles = np.linspace(np.pi* 49/100, np.pi *51/100, 100)
tested_angles = np.linspace(np.pi* 45/100, -np.pi *55/100, 100)
hspace, theta, dist, = hough_line(image, tested_angles)


h, q, d = hough_line_peaks(hspace, theta, dist)

#################################################################
#Example code from skimage documentation to plot the detected lines
angle_list=[]  #Create an empty list to capture all angles
dist_list=[]
# Generating figure 1
fig, axes = plt.subplots(1, 4, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + hspace),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), dist[-1], dist[0]],
             cmap='gray', aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')



ax[2].imshow(image, cmap='gray')
origin = np.array((0, image.shape[1]))

for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist, min_distance= 50, threshold = 0.72 * np.max(hspace))):
    angle_list.append(angle) #Not for plotting but later calculation of angles
    dist_list.append(dist)
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[2].plot(origin, (y0, y1), '-r')

ave_angle = np.mean(angle_list)
ave_dist = np.mean(dist_list)

x0, x1 = (ave_dist - origin * np.cos(ave_angle)) / np.sin(ave_angle)
ax[2].plot(origin, (x0, x1), '-b')


###############################################################
# Convert angles from radians to degrees (1 rad = 180/pi degrees)
angles = [a*180/np.pi for a in angle_list]

ax[2].set_xlim(origin)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

change = 90 - -1*np.mean(angles)

newImage = cv2.bitwise_not(imutils.rotate_bound(image, -change))
cv2.imwrite("data/image-data/Test/histogramTest.jpg", newImage)
ax[3].imshow(newImage, cmap='gray')

plt.tight_layout()
plt.show()
# Compute difference between the two lines
angle_difference = np.max(angles) - np.min(angles)
print(180 - angle_difference)   #Subtracting from 180 to show it as the small angle between two lines

# ------------------------- Histogram part -------------------------
hist = []
row_len = newImage.shape[1]
for row in newImage:
    hist.append(row_len - len(row.nonzero()[0]))

temp = []
thr = {}
c = 0
thr_num = max(hist)*0.2
for idx, p in enumerate(hist):
    if p >= thr_num and hist[idx - 1] > thr_num and idx > 0:
        temp.append(p)
        c += 1
    elif len(temp) > 0:
        thr.setdefault(idx - c, temp)
        temp = []
        c = 0

line_heights = []
thr_peaks = {}
for idx, p in enumerate(thr.items()):
    line_heights.append(p[0] + len(p[1]) - p[0])

    thr_peaks[idx] = {
        "loc": [p[0], p[0] + len(p[1])],
        "value": max(p[1]),
        "lh": p[0] + len(p[1]) - p[0]
    }

avg_lh = sum(line_heights) / len(line_heights)
q3, q1 = np.percentile(line_heights, [75, 25])
iqr = q3-q1
outlier = q1-1.5*iqr

# ----------------------- Histogram plotting -----------------------
# figure = plt.figure(figsize=(16, 12))
# fs = 25
# plt.plot(hist)
# plt.ylim(0, max(hist)*1.1)
# plt.xlabel("Row", fontsize=fs)
# plt.ylabel("Black pixels", fontsize=fs)
# plt.title("Binary image black pixel counting result", fontsize=fs)
# plt.yticks(fontsize=fs-5)
# plt.xticks(fontsize=fs-5)
# # plt.grid()
# plt.show()
# ----------------------- Histogram plotting -----------------------

figure = plt.figure(figsize=(16, 12))
fs = 25
plt.imshow(newImage)
line_heights = []
for k in thr_peaks.keys():
    for _ in thr_peaks[k].keys():
        if thr_peaks[k]["lh"] <= outlier:
            pass
        else:
            for idx, loc in enumerate(thr_peaks[k]["loc"]):
                if idx == 0:
                    plt.axhline(y=loc - avg_lh // 3, color="r", linestyle="-")
                else:
                    plt.axhline(y=loc + avg_lh // 2.5, color="r", linestyle="-")
plt.show()
# ------------------------- Histogram part -------------------------