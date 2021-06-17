import math

import cv2
from dim_reduction import get_style_char_images
from hinge_utils import *
from segmentation_to_recog import resize_pad
import matplotlib.pyplot as plt
import numpy as np
import sys

PI = 3.14159265359

# verify that this is correct
def get_angle_small(x_origin, y_origin, x_up, y_up):
    # avoid 0 division and set angle to 90 degress in that case
    if x_up - x_origin == 0:
        return 0
    else:
        val = (y_up - y_origin) / (x_up - x_origin)
    return math.atan(val)

# verify that this is correct
def get_angle(x_origin, y_origin, x_up, y_up):
    # direction vector from origin
    direction_x = x_up - x_origin
    direction_y = y_up - y_origin
    # compare this vector to horizontal vector [1, 0]
    # change of direction vector with horizontal vector
    delta_x = direction_x - 1
    phi = math.atan2(direction_y, delta_x)
    return phi

def get_histogram(list_of_cont_cords, dist_between_points, img):
    histogram = []
    i = 0
    while i < (len(list_of_cont_cords) - (2*dist_between_points)):
        # low hinge end point
        x_low = list_of_cont_cords[i][0]
        y_low = list_of_cont_cords[i][1]
        # mid point
        x_origin = list_of_cont_cords[i+dist_between_points][0]
        y_origin = list_of_cont_cords[i+dist_between_points][1]
        # 'upper' hinge end point
        x_high = list_of_cont_cords[i+(2*dist_between_points)][0]
        y_high = list_of_cont_cords[i+(2*dist_between_points)][1]
        i += dist_between_points

        # 'smaller' angle
        phi1 = get_angle_small(x_origin, y_origin, x_low, y_low)

        if not (0 <= phi1 <= PI):
            print("phi1 is a bitch")
        # 'larger' angle
        phi2 = get_angle(x_origin, y_origin, x_high, y_high)
        if not (0 <= phi2 <= 2*PI):
            print("phi2 is a bitch")

        # check if cords work
        # PLOT the hinge points
        # print(phi1, phi2)
        # hinge_cords = [(x_low, y_low), (x_origin, y_origin), (x_high, y_high)]
        # print(hinge_cords)
        # for j in range(len(hinge_cords)):
        #     img[hinge_cords[j]] = 100
        # plt.imshow(img)
        # plt.show()
        # for x in range(len(hinge_cords)):
        #     img[hinge_cords[x]] = 0
        histogram.append((phi1, phi2))

    return histogram

def get_hinge(img_label, archaic_imgs, hasmonean_imgs, herodian_imgs):

    for images in herodian_imgs[img_label]:
        images = cv2.bitwise_not(images)
        blurred_image = cv2.GaussianBlur(images, (5, 5), 0)
        Edges = cv2.Canny(blurred_image, 0, 100)
        thresh = cv2.threshold(images, 30, 255, cv2.THRESH_BINARY)[1]
        # apply canny to detect the contours of the char
        corners_of_img = cv2.Canny(thresh, 0, 100)
        cont_img = np.asarray(corners_of_img)
        plt.imshow(cont_img)
        plt.show()
        # get the coordinates of the contour pixels
        contours = np.where(cont_img == 255)
        list_of_cont_cords = list(zip(contours[0], contours[1]))
        sorted_cords = sort_cords(list_of_cont_cords)
        # plot the sorted cords in order just to be sure everything went fine
        # sorted_coords_animation(sorted_cords)
        hist = get_histogram(sorted_cords, 5, cont_img)
        print(hist)
        # hist = remove_redundant_angles(hist)

        # put the angles vals in two lists (needed to get co-occurence)
        list_phi1 = []
        list_phi2 = []
        for instance in hist:
            list_phi1.append(instance[0])
            list_phi2.append(instance[1])

        # transform entries in both lists to indices in the correspoding bin
        hist_phi1 = plt.hist(list_phi1, bins=12, range=[0, PI])
        # plt.show()
        bins_phi1 = hist_phi1[1]
        print(bins_phi1)
        inds_phi1 = np.digitize(list_phi1, bins_phi1)
        print(list_phi1)
        print(inds_phi1)
        hist_phi2 = plt.hist(list_phi2, bins=24,range=[0, 2*PI])
        # plt.show()
        bins_phi2 = hist_phi2[1]
        inds_phi2 = np.digitize(list_phi2, bins_phi2)

        hinge_features = np.zeros([12, 24], dtype=int)
        # for i in range(len(inds_phi1)):
        #     print(i)
        #     hinge_features[inds_phi1[i]][inds_phi2[i]] += 1
        # print(hinge_features)

        fig, axs = plt.subplots(2)
        fig.suptitle('Char image with histogram')
        axs[0].imshow(corners_of_img)
        axs[1].hist2d(inds_phi1, inds_phi2, bins=[12, 24])
        plt.show()




#---TO-DO:---#
#1)create def get_textural_feature
#2) get image, get image label from CNN
#3) create SOM:
#   3)a)get all specific characters from three periods given the CNN label
#   4)b) extract allographic and textural features from all these characters
#   4)c) apply PCA on vectors from 4)b) (if needed)
#   ..

if __name__ == '__main__':
    # / home / jan / PycharmProjects / HandwritingRecog /
    style_base_path = '/home/jan/PycharmProjects/HandwritingRecog/data/characters_for_style_classification_balance_morph/'
    style_archaic_path = style_base_path + 'Archaic/'
    style_hasmonean_path = style_base_path + 'Hasmonean/'
    style_herodian_path = style_base_path + 'Herodian/'

    archaic_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Nun-final', 'He', 'Qof', 'Kaf', 'Samekh', 'Yod', 'Dalet',
                        'Waw', 'Ayin', 'Mem', 'Gimel', 'Bet', 'Shin', 'Resh', 'Alef', 'Het']
    hasmonean_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                            'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final', 'Gimel',
                            'Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het']
    herodian_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                        'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final', 'Gimel',
                        'Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het']

    # Retrieve img lists from each class' each character AND resize them
    new_size = (40, 40)  # change this to something which is backed up by a reason
    archaic_imgs = {char:
                    [cv2.resize(img, new_size).flatten() for img in get_style_char_images(style_archaic_path, char)]
                    for char in archaic_characters}
    hasmonean_imgs = {char:
                    [cv2.resize(img, new_size).flatten() for img in get_style_char_images(style_hasmonean_path, char)]
                    for char in hasmonean_characters}
    herodian_imgs = {char:
                    [cv2.resize(img, new_size).flatten() for img in get_style_char_images(style_herodian_path, char)]
                    for char in herodian_characters}

    print("working")
    archaic_imgs_unflattened = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_archaic_path, char)]
                    for char in archaic_characters}
    hasmonean_imgs_unflattened = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_hasmonean_path, char)]
                    for char in hasmonean_characters}
    herodian_imgs_unflattened = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_herodian_path, char)]
                    for char in herodian_characters}

    #show img
    #imgplot = plt.imshow(herodian_imgs['Alef'][0])
    #plt.show()

    img_label = 'Alef'
    #get hinge histogram
    hingehist_archaic,hingehist_hasmonean,hingehist_herodian = get_hinge(img_label,archaic_imgs_unflattened,hasmonean_imgs_unflattened,herodian_imgs_unflattened)

