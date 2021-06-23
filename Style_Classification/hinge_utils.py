import numpy as np
import matplotlib.pyplot as plt
import cv2

def noise_removal(img,morphology=False):
    #remove unneeded nosie from image
    img = cv2.bitwise_not(img)
    resized_pad_img = img.copy()
    # Filter using contour area and remove small noise
    retval, resized_pad_img = cv2.threshold(resized_pad_img.copy(), thresh=30, maxval=255,
                                   type=cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(resized_pad_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
    # ROI will be object with biggest contour
    cnt = cnt[1:]
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    for c in cnt:
        cv2.drawContours(mask, [c], -1, 0, -1)
    mask = cv2.bitwise_not(mask)
    newimg = (resized_pad_img - mask) 
    newimg = cv2.bitwise_not(newimg)
    #apply morphological operations if neede to remove excess noise
    if morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        newimg = cv2.morphologyEx(newimg, cv2.MORPH_CLOSE, kernel)
    return newimg,mask

# finds closest cord based on Manhatten distance with an added prio to the x-direction
def find_closest_cord(current_cord, contour_cords):
    min_dist = 500
    min_dist_x = 500
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
            dist_x = abs(current_cord[0]-contour_cords[j][0])
            if dist_x < min_dist_x:
                min_dist_x = dist_x
                closest_cord = contour_cords[min_dist_list[j][0]]

    return closest_cord

# LOGIC
# start at one cord then always find the closest in x,y while prioritizing one direction
# (so if x_next - x == y_next - y always choose either x or y)
# and add that to the new list, when we add to new list remove from old
# so we cannot go back // stop when list of contour cords is empty
def sort_cords(contour_cords):
    sorted_list = []
    current_cord = contour_cords[0]
    sorted_list.append(current_cord)
    contour_cords.remove(current_cord)
    while contour_cords != []:
        closest_cord = find_closest_cord(current_cord, contour_cords)
        sorted_list.append(closest_cord)
        contour_cords.remove(closest_cord)
        current_cord = closest_cord
    return sorted_list

def get_chisquared(feature_vector,style_pdf):
    chi = 0
    for i in range(len(feature_vector)):
        if feature_vector[i] == 0 and style_pdf[i] == 0 :
            continue
        chi += (feature_vector[i]-style_pdf[i])**2 / (feature_vector[i]+style_pdf[i])
        
    return chi/2
    

# remove all occurences where phi2 < phi1
def remove_redundant_angles(hist):
    for instance in hist:
        if instance[0] <= instance[1]:
            hist.remove(instance)
    return hist

def sorted_coords_animation(sorted_cords, hinge_coords=None):
    dummy_img = np.zeros([40, 40])
    for i in range(len(sorted_cords)):
        print(sorted_cords[i])
        dummy_img[sorted_cords[i]] = 1
        plt.imshow(dummy_img)
        plt.pause(0.0001)

    if hinge_coords is not None:
        for j in range(len(hinge_coords)):
            print(hinge_coords[j])
            dummy_img[hinge_coords[j]] = 0.5
