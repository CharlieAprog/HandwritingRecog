import numpy as np
import matplotlib.pyplot as plt

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

# remove all occurences where phi2 < phi1
def remove_redundant_angles(hist):
    print(len(hist))
    for instance in hist:
        if instance[0] <= instance[1]:
            hist.remove(instance)
    print(len(hist))
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
