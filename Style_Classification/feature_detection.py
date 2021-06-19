import math
import cv2
from dim_reduction import get_style_char_images
from hinge_utils import *
#from segmentation_to_recog import resize_pad
import matplotlib.pyplot as plt
import numpy as np
import sys

PI = 3.14159265359


def get_angle(x_origin, y_origin, x_up, y_up):
    output = math.degrees(math.atan2(y_up-y_origin,x_up-x_origin))

    if output < 0:
        output = 180 + (180+output)

    return output

def get_histogram(list_of_cont_cords, dist_between_points, img):
    histogram = []
    i = 0
    while i < (len(list_of_cont_cords) - (2*dist_between_points)):
        # low hinge end point
        x_low = list_of_cont_cords[i][1]
        y_low = list_of_cont_cords[i][0]
        # mid point
        x_origin = list_of_cont_cords[i+dist_between_points][1]
        y_origin = list_of_cont_cords[i+dist_between_points][0]
        # 'upper' hinge end point
        x_high = list_of_cont_cords[i+(2*dist_between_points)][1]
        y_high = list_of_cont_cords[i+(2*dist_between_points)][0]
        i += dist_between_points

        phi1 = get_angle(x_origin, (40-y_origin), x_low, (40-y_low))

        phi2 = get_angle(x_origin, (40-y_origin), x_high, (40-y_high))

        histogram.append((phi1, phi2))
        # check if cords work
        # PLOT the hinge points
        # print(phi1, phi2)
        # hinge_cords = [(y_low, x_low), (y_origin, x_origin), (y_high,x_high)]
        # print(hinge_cords)
        # for j in range(len(hinge_cords)):
        #     img[hinge_cords[j]] = 100
        # plt.imshow(img)
        # plt.show()
        # for x in range(len(hinge_cords)):
        #     img[hinge_cords[x]] = 0
        # histogram.append((phi1, phi2))

    return histogram

def hinge_main(img_label,imgs):
    for images in imgs[img_label]:
            images = cv2.GaussianBlur(images,(5,5),0)
            img,mask = noise_removal(images)
            fig, axs = plt.subplots(3)
            axs[0].imshow(images)
            axs[1].imshow(img)
            axs[2].imshow(mask)
            plt.show()
            # apply canny to detect the contours of the char
            corners_of_img = cv2.Canny(img, 0, 100)
            cont_img = np.asarray(corners_of_img)

            # get the coordinates of the contour pixels
            contours = np.where(cont_img == 255)
            list_of_cont_cords = list(zip(contours[0], contours[1]))
            sorted_cords = sort_cords(list_of_cont_cords)

            # plot the sorted cords in order just to be sure everything went fine
            # sorted_coords_animation(sorted_cords)

            hist = get_histogram(sorted_cords, 2, cont_img)
            print(hist)
            print('-----|----')
            #if phi2>=phi1, remove from hist
            hist = remove_redundant_angles(hist)
            print(hist)
            # put the angles vals in two lists (needed to get co-occurence)
            list_phi1 = []
            list_phi2 = []
            for instance in hist:
                list_phi1.append(instance[0])
                list_phi2.append(instance[1])

            # transform entries in both lists to indices in the correspoding bin
            hist_phi1 = plt.hist(list_phi1, bins=24, range=[0, 360])
            # plt.show()

            bins_phi1 = hist_phi1[1]
            inds_phi1 = np.digitize(list_phi1, bins_phi1)

            hist_phi2 = plt.hist(list_phi2, bins=24,range=[0, 360])
            # plt.show()
            bins_phi2 = hist_phi2[1]
            inds_phi2 = np.digitize(list_phi2, bins_phi2)

            hinge_features = np.zeros([24, 24], dtype=int)
            
            num_features = 0
            for i in range(len(inds_phi1)):
                if inds_phi1[i] <= inds_phi2[i]:
                    num_features += 1
                    hinge_features[inds_phi1[i]-1][inds_phi2[i]-1] += 1
            print(num_features,'num_features')
            print(hinge_features,'hinge_features')

            feature_vector = []
            x= 0
            for j in range(24):
                feature_vector.append(hinge_features[j][x:])
                x += 1
            feature_vector = [item for sublist in feature_vector for item in sublist]
            print(feature_vector, len(feature_vector))
            print(sum(feature_vector))
            notalistbitch = np.asarray(feature_vector)
            massdist = [element/sum(feature_vector) for element in notalistbitch]
            print(massdist)
            plt.show()
            vals = np.arange(0, 300)

            fig, axs = plt.subplots(2)
            fig.suptitle('Char image with histogram')
            axs[0].imshow(corners_of_img)
            axs[1].hist2d(inds_phi1, inds_phi2, bins=[12, 24])
            plt.show()
    return  hinge_features

def get_hinge(img_label, archaic_imgs, hasmonean_imgs, herodian_imgs):

    archaic_hinge = hinge_main(img_label,archaic_imgs)
    hasmonean_hinge = hinge_main(img_label,hasmonean_imgs)
    herodian_hinge = hinge_main(img_label,herodian_imgs)

    print(archaic_hinge)
    print(hasmonean_hinge)
    print(herodian_hinge)
    return archaic_hinge,hasmonean_hinge,herodian_hinge



#---TO-DO:---#
#1)create def get_textural_feature
#2) get image, get image l
# abel from CNN
#3) create SOM:
#   3)a)get all specific characters from three periods given the CNN label
#   4)b) extract allographic and textural features from all these characters
#   4)c) apply PCA on vectors from 4)b) (if needed)
#   ..

if __name__ == '__main__':
    # / home / jan / PycharmProjects / HandwritingRecog /
    style_base_path = 'C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/data/characters_for_style_classification_balance_morph/'
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

    print("working")
    archaic_imgs = {char:
                    [ cv2.resize(img, new_size) for img in get_style_char_images(style_archaic_path, char)]
                    for char in archaic_characters}
    hasmonean_imgs = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_hasmonean_path, char)]
                    for char in hasmonean_characters}
    herodian_imgs = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_herodian_path, char)]
                    for char in herodian_characters}

    
    img_label = 'Lamed'
    #get hinge histogram
    hingehist_archaic,hingehist_hasmonean,hingehist_herodian = get_hinge(img_label,archaic_imgs,hasmonean_imgs,herodian_imgs)

