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

def get_histogram(list_of_cont_cords, dist_between_points, img):
    # get histogram of co-occurences
    histogram = []
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
        i += dist_between_points

        phi1 = get_angle(x_origin, (40 - y_origin), x_low, (40 - y_low))
        phi2 = get_angle(x_origin, (40 - y_origin), x_high, (40 - y_high))

        histogram.append((phi1, phi2))
    return histogram


def get_hinge_pdf(img_label, imgs):
    vals = [i * 0 for i in range(300)]
    for images in imgs[img_label]: #for all Global characters  
            #removenoise + gaussian blur for better canny edge detection
            images = cv2.GaussianBlur(images,(5,5),0)
            img,mask = noise_removal(images)

            # apply canny to detect the contours of the char
            corners_of_img = cv2.Canny(img, 0, 100)
            cont_img = np.asarray(corners_of_img)

            # get the coordinates of the contour pixels
            contours = np.where(cont_img == 255)
            list_of_cont_cords = list(zip(contours[0], contours[1]))
            sorted_cords = sort_cords(list_of_cont_cords)

            # plot the sorted cords in order just to be sure everything went fine
            # sorted_coords_animation(sorted_cords)

            hist = get_histogram(sorted_cords, 5, cont_img)
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
            
            feature_vector = np.asarray(feature_vector)
            vals = vals+feature_vector
    
    npvector = np.asarray(vals)
    massdist = [element / sum(vals) for element in npvector]

    return massdist

def get_char_vector(img):
    #returns pdf of hinge features (f2) for one image
    # img,mask = noise_removal(img)
    # apply canny to detect the contours of the char
    img[img==1]=255
    # plt.imshow(img)
    # plt.show()
    corners_of_img = cv2.Canny(img, 0, 100)
    cont_img = np.asarray(corners_of_img)
    # get the coordinates of the contour pixels
    contours = np.where(cont_img == 255)
    list_of_cont_cords = list(zip(contours[0], contours[1]))
    sorted_cords = sort_cords(list_of_cont_cords)

    # plot the sorted cords in order just to be sure everything went fine
    # sorted_coords_animation(sorted_cords)

    #get histogram of phi's
    hist = get_histogram(sorted_cords, 5, cont_img)

    #if phi2>=phi1, remove from hist
    # hist = remove_redundant_angles(hist)

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
        if inds_phi1[i] <= inds_phi2[i]:
            num_features += 1
            hinge_features[inds_phi1[i] - 1][inds_phi2[i] - 1] += 1
    feature_vector = []
    x = 0
    for j in range(24):
        feature_vector.append(hinge_features[j][x:])
        x += 1

    feature_vector = [item for sublist in feature_vector for item in sublist]
    feature_vector = np.asarray(feature_vector)

    #return pdf
    return [element/sum(feature_vector) for element in feature_vector]