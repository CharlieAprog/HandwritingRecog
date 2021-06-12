import math

import cv2
from dim_reduction import get_style_char_images
from segmentation_to_recog import resize_pad
import matplotlib.pyplot as plt
import numpy as np
import sys

PI = 3.14159265359
# def get_angle(x_low, y_low, x_up, y_up):
#     dir_vector_x = x_up - x_low
#     dir_vector_y = y_up - y_low
#     y_inp = -dir_vector_x
#     x_inp = dir_vector_y
#     phi = math.atan2(y_inp, x_inp)
#     return phi
def get_angle(x_low, y_low, x_up, y_up):
    val = (y_up - y_low) / (x_up-x_low)
    phi = math.atan(val)
    return phi

def get_histogram(list_of_cont_cords, dist_between_points):
    histogram = []
    i = 0
    while i < (len(list_of_cont_cords) - (2*dist_between_points)):
        x_low = list_of_cont_cords[i][0]
        y_low = list_of_cont_cords[i][1]
        x_center = list_of_cont_cords[i+dist_between_points][0]
        y_center = list_of_cont_cords[i + dist_between_points][1]
        x_high = list_of_cont_cords[i+(2*dist_between_points)][0]
        y_high = list_of_cont_cords[i + (2 * dist_between_points)][1]
        i += 1
        # avoid rare 0 division
        if (x_center - x_low) != 0:
            # 'smaller' angle
            phi1 = get_angle(x_low, y_low, x_center, y_center)
            # rescale tp [0, pi]
            phi1 += PI / 2
        if (x_high - x_center) != 0:
            # 'larger' angle
            phi2 = get_angle(x_center, y_center, x_high, y_high)
            # rescale to [0, 2pi]
            phi2 += PI / 2
            phi2 = 2*PI - phi2

        histogram.append((phi1, phi2))

    return histogram

def get_hinge(img_label, archaic_imgs, hasmonean_imgs, herodian_imgs):
    # given a time period, calculate the Hinge histogram occurences of phi1 and phi2
    '''
    sample img to develop function
    sample_img = herodian_imgs[img_label][0]
    Gaussian blurring w/ 5x5 kernel for better edge detection
    blurred_image  = cv2.GaussianBlur(img,(5,5),0)
    Edges = cv2.Canny(blurred_image,0,100)
    show edges found
    imgplot = plt.imshow(Edges)
    plt.show()
    '''
    for images in herodian_imgs[img_label]:
        images = cv2.bitwise_not(images)
        # print(images)
        blurred_image = cv2.GaussianBlur(images, (5, 5), 0)
        Edges = cv2.Canny(blurred_image, 0, 100)
        thresh = cv2.threshold(images, 30, 255, cv2.THRESH_BINARY)[1]
        # apply canny to detect the contours of the char
        corners_of_img = cv2.Canny(thresh, 0, 100)
        cont_img = np.asarray(corners_of_img)
        # get the coordinates of the contour pixels
        contours = np.where(cont_img == 255)
        list_of_cont_cords = list(zip(contours[0], contours[1]))


        hist = get_histogram(list_of_cont_cords, 5)
        x_plt = []
        y_plt = []
        for i in range(len(hist)):
            x_plt.append(hist[i][0])
            y_plt.append(hist[i][1])

        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].imshow(corners_of_img)
        axs[1].hist2d(x_plt,y_plt, bins = [15, 15])
        plt.show()


        # corners = cv2.goodFeaturesToTrack(thresh, 4, 0.01, 50, useHarrisDetector=True, k=0.04)
        # imagesrgb = cv2.cvtColor(images, cv2.COLOR_GRAY2RGB)
        # print("Corners:")
        # for c in corners:
        #     x, y = c.ravel()
        #     print(corners)
        #     cv2.circle(imagesrgb, (x, y), 3, (255, 0,), -1)
        #
        # cv2.imshow("binary image", thresh)
        # cv2.imshow("iamge/cornersdrawn", imagesrgb)
        # cv2.waitKey(0)

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
    new_size = (50, 50)  # change this to something which is backed up by a reason
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

