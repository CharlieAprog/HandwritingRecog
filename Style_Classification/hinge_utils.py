import numpy as np
import matplotlib.pyplot as plt
import cv2
from Style_Classification.Classify_Char_Style import *


'''
Helper function to remove noise/artefacts from an Image
'''
def noise_removal(img,morphology=False):
    #remove unneeded noise from image
    img = cv2.bitwise_not(img)
    resized_pad_img = img.copy()

    # Filter using contour area and remove small noise
    retval, resized_pad_img = cv2.threshold(resized_pad_img.copy(), thresh=30, maxval=255,
                                   type=cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(resized_pad_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)

    #create and get masks with noise contours
    cnt = cnt[1:]
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    for c in cnt:
        cv2.drawContours(mask, [c], -1, 0, -1)
    mask = cv2.bitwise_not(mask)

    #apply masking to image
    newimg = (resized_pad_img - mask) 
    newimg = cv2.bitwise_not(newimg)

    #apply morphological operations if neede to remove excess noise
    if morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        newimg = cv2.morphologyEx(newimg, cv2.MORPH_CLOSE, kernel)
    return newimg,mask

'''
Finds the clostest coordinate in a List based on Manhatten distance.
Has a priority for finding coords in the y-direction.
'''
def find_closest_cord(current_cord, contour_cords):

    min_dist = 5000
    min_dist_x = 5000
    min_dist_list = []

    # get the cords that are at min_dist from the current cord
    for i in range(len(contour_cords)):
        dist = abs(current_cord[0]-contour_cords[i][0]) + abs(current_cord[1]-contour_cords[i][1])

        if dist <= min_dist:
            min_dist = dist
            min_dist_list.append((i, min_dist))

    # from the cords that have an equal min dist get the one closest in terms of y
    for j in range(len(min_dist_list)):
        if min_dist_list[j][1] == min_dist:

            # add direction prio
            dist_y = abs(current_cord[1]-contour_cords[j][1])
            dist_x = abs(current_cord[0]-contour_cords[j][0])
            # set prio in special cases

            if dist_x == 1 and dist_y == 1:
                dist_x = -0.1
            if dist_x == 1 and dist_y == 0:
                dist_x = -0.2
            if dist_x == 0 and dist_y == 1:
                dist_x = -0.3

            if dist_x < min_dist_x:
                min_dist_x = dist_x
                closest_cord = contour_cords[min_dist_list[j][0]]

    return closest_cord, min_dist

'''
Sorts the coordinates of a given contour, so that hinge points can be found.
LOGIC
start at one cord then always find the closest in x,y while prioritizing one direction
(so if x_next - x == y_next - y always choose either x or y)
and add that to the new list, when we add to new list remove from old
so we cannot go back // stop when list of contour cords is empty
'''
def sort_cords(contour_cords):
    sorted_list = []
    connected_contour = []
    current_cord = contour_cords[0]
    connected_contour.append(current_cord)
    contour_cords.remove(current_cord)
    len_contour_list = len(contour_cords)

    while len(contour_cords) != 0:
        closest_cord, min_dist = find_closest_cord(current_cord, contour_cords)
        # if the next closest cord is more then 2 distance we encounter a 'new' connected component
        # to make the angles work these are only calculated on these connected components

        if min_dist < 3:
            connected_contour.append(closest_cord)
            contour_cords.remove(closest_cord)
            current_cord = closest_cord

        else:
            sorted_list.append(connected_contour)
            connected_contour = []
            connected_contour.append(closest_cord)
            contour_cords.remove(closest_cord)
            current_cord = closest_cord

    # check if we used all the cords and otherwise append the final connected_contours
    total_len = 0
    for contour in sorted_list:
        total_len += len(contour)
    if total_len != len_contour_list:
        sorted_list.append(connected_contour)


    return sorted_list

def get_chisquared(feature_vector,style_pdf):
    #get chi square
    chi = 0
    for i in range(len(feature_vector)):
        if feature_vector[i] == 0 and style_pdf[i] == 0 :
            continue
        chi += (feature_vector[i]-style_pdf[i])**2 / (feature_vector[i]+style_pdf[i])
        
    return chi/2
    
'''
Helper function to plot sorted coordinates in order,
creates a little animation.
''' 
def sorted_coords_animation(sorted_cords, hinge_coords=None):
    dummy_img = np.zeros([40, 40])

    for cords in sorted_cords:
        print(cords)
        for i in range(len(cords)):
            
            dummy_img[cords[i]] = 1
            plt.imshow(dummy_img)
            plt.pause(0.0001)


