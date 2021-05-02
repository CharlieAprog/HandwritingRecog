from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import PIL
from PIL import Image
import PIL.ImageOps 

import cv2
from matplotlib import pyplot as plt

img_path = 'data/image-data/binary/P21-Fg006-R-C01-R01-binarized.jpg'
#image = Image.open(img_path)
#image = PIL.ImageOps.invert(image)
image = cv2.imread(img_path, 0)
image = cv2.bitwise_not(image)
plt.imshow(image, cmap='gray')
plt.show()

tested_angles = np.linspace(-np.pi/2, np.pi/2, 180)
hspace, theta, dist, = hough_line(image, tested_angles)

plt.figure()
plt.imshow(hspace)
h, q, d = hough_line_peaks(hspace, theta, dist)

#################################################################
#Example code from skimage documentation to plot the detected lines
angle_list=[]  #Create an empty list to capture all angles

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
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

for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist)):
    angle_list.append(angle) #Not for plotting but later calculation of angles
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[2].plot(origin, (y0, y1), '-r')
ax[2].set_xlim(origin)
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()

###############################################################
# Convert angles from radians to degrees (1 rad = 180/pi degrees)
angles = [a*180/np.pi for a in angle_list]

# Compute difference between the two lines
angle_difference = np.max(angles) - np.min(angles)
print(180 - angle_difference)   #Subtracting from 180 to show it as the small angle between two lines

