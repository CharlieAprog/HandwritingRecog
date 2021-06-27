import cv2
import matplotlib.pyplot as plt

from Style_Classification.feature_detection import *



def get_style_char_images(style_path: str, character: str):
    """ Returns a list of numpy arrays, where each arrays corresponds to an image from a certain character of a
    certain class """
    if character == 'Global':
        list_for_glob = style_path + '*/*.jpg'
    else:
        list_for_glob = style_path + character + '/*.jpg'
    img_name_list = glob.glob(list_for_glob)
    img_list = [cv2.imread(img, 0) for img in img_name_list]
    assert len(img_list) > 0, "Trying to read image files while being in a wrong folder."
    return img_list

def get_angle(x_origin, y_origin, x_up, y_up):
    #get angle for hinge pdf (ph1 or phi2)
    output = math.degrees(math.atan2(y_up-y_origin,x_up-x_origin))
    if output < 0:
        output = 180 + (180 + output)

    return output

def get_histogram(list_of_contours, dist_between_points, img, show_points=False):
    # get histogram of co-occurences
    histogram = []
    hinge_points= []
    # go thru every connected contour and extract the angles
    for list_of_cont_cords in list_of_contours:
        i = 0
        while i < (len(list_of_cont_cords) - (2 * dist_between_points)):
            # low hinge end point
            x_low = list_of_cont_cords[i][1]
            y_low = list_of_cont_cords[i][0]
            # mid point
            x_origin = list_of_cont_cords[i + dist_between_points][1]
            y_origin = list_of_cont_cords[i + dist_between_points][0]
            # 'upper' hinge end point
            x_high = list_of_cont_cords[i + (2 * dist_between_points)][1]
            y_high = list_of_cont_cords[i + (2 * dist_between_points)][0]


            distance_low_leg = abs(y_low - y_origin) + abs(x_low - x_origin)
            distance_high_leg = abs(y_high - y_origin) + abs(x_high - x_origin)
            if ((dist_between_points-1 <= distance_low_leg <= dist_between_points+1)
                and (dist_between_points-1 <= distance_high_leg <= dist_between_points+1)):
                phi1 = get_angle(x_origin, (40 - y_origin), x_low, (40 - y_low))
                phi2 = get_angle(x_origin, (40 - y_origin), x_high, (40 - y_high))

                histogram.append((phi1, phi2))
                i += dist_between_points
                if show_points:
                    hinge_points.append([(y_low, x_low), (y_origin, x_origin), (y_high, x_high)])
            else:
                i += 1
    if show_points:
        for points in hinge_points:
            for j in range(len(points)):
                img[points[j]] = 150
            plt.imshow(img, cmap='gray')
            plt.show()
            for j in range(len(points)):
                img[points[j]] = 255

    return histogram


def get_hinge_pdf(img_label, imgs):
    vals = [i * 0 for i in range(300)]
    for images in imgs[img_label]: #for all Global characters  
            #removenoise + gaussian blur for better canny edge detection
            # retval,images = cv2.threshold(images.copy(), thresh=100, maxval=255,
            #                        type=cv2.THRESH_BINARY_INV)
            # print(';sup')
            # plt.imshow(images)
            # plt.show()
            img = cv2.GaussianBlur(images,(5,5),0)
            img,mask = noise_removal(images)


            # apply canny to detect the contours of the char
            corners_of_img = cv2.Canny(img, 0, 100)
            cont_img = np.asarray(corners_of_img)
            # plt.show()
            # plt.imshow(corners_of_img)
            # plt.show()
            # get the coordinates of the contour pixels
            contours = np.where(cont_img == 255)
            list_of_cont_cords = list(zip(contours[0], contours[1]))
            sorted_cords = sort_cords(list_of_cont_cords)

            # plot the sorted cords in order just to be sure everything went fine
            # sorted_coords_animation(sorted_cords)

            hist = get_histogram(sorted_cords, 3, cont_img)
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

            feature_vector = []
            x= 0
            for j in range(24):
                feature_vector.append(hinge_features[j][x:])
                x += 1
            feature_vector = [item for sublist in feature_vector for item in sublist]
            if len(hist) > 25:
                feature_vector = np.asarray(feature_vector)
                vals = vals+feature_vector
    
    npvector = np.asarray(vals)
    massdist = [element / sum(vals) for element in npvector]

    return massdist

def get_char_vector(img, image_from_page=True):
    #returns pdf of hinge features (f2) for one image
    # img,mask = noise_removal(img)
    # apply canny to detect the contours of the char
    img = np.uint8(img)
    if image_from_page:
        img[img == 1] = 255
    else:
        img[img == 1] = 255
        # retval, img = cv2.threshold(img.copy(), thresh=100, maxval=255,
        #                                type=cv2.THRESH_BINARY_INV)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img, mask = noise_removal(img)
    # #img = cv2.GaussianBlur(img, (5, 5), 0)

    corners_of_img = cv2.Canny(img, 0, 100)
    # plt.show()
    # plt.imshow(corners_of_img)
    # plt.show()
    # plt.show()
    # plt.imshow(corners_of_img)
    # plt.show()
    cont_img = np.asarray(corners_of_img)
    # get the coordinates of the contour pixels
    contours = np.where(cont_img == 255)
    list_of_cont_cords = list(zip(contours[0], contours[1]))

    sorted_cords = sort_cords(list_of_cont_cords)
    # plot the sorted cords in order just to be sure everything went fine
    # sorted_coords_animation(sorted_cords)

    #get histogram of phi's
    hist = get_histogram(sorted_cords, 3, cont_img, show_points=False)

    # put the angles vals in two lists (needed to get co-occurence)
    list_phi1 = []
    list_phi2 = []
    for instance in hist:
        list_phi1.append(instance[0])
        list_phi2.append(instance[1])

    # transform entries in both lists to indices in the correspoding bin
    hist_phi1 = plt.hist(list_phi1, bins=24, range=[0, 360])
    bins_phi1 = hist_phi1[1]
    inds_phi1 = np.digitize(list_phi1, bins_phi1)
    hist_phi2 = plt.hist(list_phi2, bins=24, range=[0, 360])
    bins_phi2 = hist_phi2[1]
    inds_phi2 = np.digitize(list_phi2, bins_phi2)

    hinge_features = np.zeros([24, 24], dtype=int)
    num_features = 0
    for i in range(len(inds_phi1)):
        # ignore redundant angles
        if inds_phi1[i] <= inds_phi2[i]:
            num_features += 1
            hinge_features[inds_phi1[i] - 1][inds_phi2[i] - 1] += 1
    feature_vector = []
    # only keep upper diagonal of co-occurence matrix
    x = 0
    for j in range(24):
        feature_vector.append(hinge_features[j][x:])
        x += 1

    feature_vector = [item for sublist in feature_vector for item in sublist]
    feature_vector = np.asarray(feature_vector)
    if len(hist) > 25:
        #return pdf
        return [element/sum(feature_vector) for element in feature_vector]
    else:
        return 0