import math
import cv2
from skimage import feature
from Style_Classification.hinge_utils import *
# from segmentation_to_recog import resize_pad
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats
from sklearn.decomposition import PCA
import glob
from collections import Counter

PI = 3.14159265359


def get_style_char_images(style_path: str, character: str):
    """ Returns a list of numpy arrays, where each arrays corresponds to an image from a certain character of a
    certain class """
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
        #delete them when code's ready
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


def get_hinge_pdf(img_label, imgs):
    vals = [i * 0 for i in range(300)]
    for images in imgs[img_label]:
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
            #if phi2>=phi1, remove from hist
            hist = remove_redundant_angles(hist)
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
            # print(vals, len(vals))
            # print(sum(vals))
            # fig, axs = plt.subplots(2)
            # fig.suptitle('Char image with histogram')
            # axs[0].imshow(corners_of_img)
            # axs[1].plot(vals, massdist)
    
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
    hist = remove_redundant_angles(hist)

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

def get_style_char_vec(characters,labels):
    #main pipeline function to get char
    style_char_vec = []
    chi_squared_vec = []
    style_base_path = 'data/characters_for_style_classification_balance_morph/'
    
    archaic_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Nun-final', 'He', 'Qof', 'Kaf', 'Samekh', 'Yod', 'Dalet',
                          'Waw', 'Ayin', 'Mem', 'Gimel', 'Bet', 'Shin', 'Resh', 'Alef', 'Het']

    hasmonean_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                            'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                            'Gimel',
                            'Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het']

    herodian_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                           'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                           'Gimel',
                           'Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het']

    # Retrieve img lists from each class' each character AND resize them
    new_size = (50, 50)  # change this to something which is backed up by a reason

#get image dict
    archaic_imgs = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_base_path +'Archaic/' , char)]
                    for char in archaic_characters}
    hasmonean_imgs ={char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_base_path +'Hasmonean/', char)]
                    for char in hasmonean_characters}
    herodian_imgs = {char:
                     [cv2.resize(img, new_size) for img in get_style_char_images(style_base_path +'Herodian/', char)]
                     for char in herodian_characters}
    idx2name = {0: 'Alef', 1: 'Ayin', 2: 'Bet', 3:'Dalet', 4: 'Gimel', 5:'He',
            6: 'Het', 7: 'Kaf', 8: 'Kaf-final', 9:'Lamed', 10: 'Mem',
            11: 'Mem-medial', 12: 'Nun-final', 13: 'Nun-medial', 14: 'Pe',
            15: 'Pe-final', 16: 'Qof', 17: 'Resh', 18: 'Samekh', 19: 'Shin',
            20: 'Taw', 21: 'Tet', 22: 'Tsadi-final', 23: 'Tsadi-medial',
            24: 'Waw', 25: 'Yod', 26: 'Zayin'}
    
    archaic_pdfs={}
    hasmonean_pdfs ={}
    herodian_pdfs = {}

    for image,label in zip(characters,labels):
        if (label == 21 or label == 22 or label == 13 or label == 11
                    or label == 15 or label ==  26 or label == 23):
            if idx2name[label] not in hasmonean_pdfs: 
                hasmonean_pdfs[idx2name[label]] = get_hinge_pdf(idx2name[label], hasmonean_imgs)
            if idx2name[label] not in herodian_pdfs: 
                herodian_pdfs[idx2name[label]] = get_hinge_pdf(idx2name[label], herodian_imgs)    
                
            # get feature vector of given char and get chisquared for each pdf 
            feature_vector = get_char_vector(image)   
            chihasmonean = get_chisquared(feature_vector,hasmonean_pdfs[idx2name[label]])
            chiherodian = get_chisquared(feature_vector,herodian_pdfs[idx2name[label]])
            minchi = min(chihasmonean,chiherodian)
                    
            if minchi == chihasmonean: predicted = 'Hasmonean' 
            else: predicted = 'Herodian'
            style_char_vec.append(predicted)
            chi_squared_vec.append(minchi)

        else:
            #get hinge pdfs
            if idx2name[label] not in archaic_pdfs: 
                archaic_pdfs[idx2name[label]] = get_hinge_pdf(idx2name[label],archaic_imgs)
            if idx2name[label] not in hasmonean_pdfs: 
                hasmonean_pdfs[idx2name[label]] = get_hinge_pdf(idx2name[label],hasmonean_imgs)
            if idx2name[label] not in herodian_pdfs: 
                herodian_pdfs[idx2name[label]] = get_hinge_pdf(idx2name[label],herodian_imgs)

            #calculate vector for char and chisquared distance
            feature_vector = get_char_vector(image)            
            chiarchaic = get_chisquared(feature_vector,archaic_pdfs[idx2name[label]])
            chihasmonean = get_chisquared(feature_vector,hasmonean_pdfs[idx2name[label]])
            chiherodian = get_chisquared(feature_vector,herodian_pdfs[idx2name[label]])

            minchi = min(chihasmonean,chiherodian,chiarchaic)
            #smallest chi squared is the style of char
            if minchi == chiarchaic: predicted ='Archaic'
            if minchi == chihasmonean: predicted = 'Hasmonean'
            if minchi ==chiherodian: predicted = 'Herodian'
            style_char_vec.append(predicted)
            chi_squared_vec.append(minchi)

    return style_char_vec,chi_squared_vec

def get_dominant_style(style_vec,chisquared_vec,n_neighbors = 10):
    style_vec = [sorting for _, sorting in sorted(zip(chisquared_vec, style_vec))]    
    return   Counter(style_vec[:n_neighbors])


def get_accuracy_alldata(dataset,archaic_imgs,hasmonean_imgs,herodian_imgs):
    ##Get accuracy: for finetuning and testing purposes
    cor = 0
    for stylename,styledataset in dataset.items():
        for label,characterset in styledataset.items():
            #arhaic dataset doesnt have theselabels: we only label these as hasmonena & herodian
            if (label == 'Tet' or label == 'Tsadi-final' or label == 'Nun-medial' or label == 'Mem-medial'
                    or label == 'Pe-final' or label == 'Zayin' or label == 'Tsadi-medial'):
                hasmonean_pdf = get_hinge_pdf(label, hasmonean_imgs)
                herodian_pdf = get_hinge_pdf(label, herodian_imgs)
                for image in characterset:
                    # get feature vector of given char and get chisquared for each pdf 
                    feature_vector = get_char_vector(image)   
                    chihasmonean = get_chisquared(feature_vector,hasmonean_pdf)
                    chiherodian = get_chisquared(feature_vector,herodian_pdf)
                    minchi = min(chihasmonean,chiherodian)
                    
                    #classify and debug etc
                    if stylename == 'hasmonean' and minchi == chihasmonean:
                        cor+=1
                    if stylename =='herodian' and minchi == chiherodian:
                        cor+=1
                    
                    # if minchi == chihasmonean: predicted = 'hasmonean'
                    # if minchi ==chiherodian: predicted = 'herodian'
                    # if stylename != predicted :
                    #     print(' wrong classification')
                    #     print('true:',stylename)
                    #     print('label:',label)
                    #     print('predicted:',predicted)
            else:
                #get hinge pdfs
                archaic_pdf = get_hinge_pdf(label,archaic_imgs)
                hasmonean_pdf = get_hinge_pdf(label,hasmonean_imgs)
                herodian_pdf = get_hinge_pdf(label,herodian_imgs)

                for image in characterset:
                    #calculate vector for char and chisquared distance
                    feature_vector = get_char_vector(image)            
                    chiarchaic = get_chisquared(feature_vector,archaic_pdf)
                    chihasmonean = get_chisquared(feature_vector,hasmonean_pdf)
                    chiherodian = get_chisquared(feature_vector,herodian_pdf)

                    minchi = min(chihasmonean,chiherodian,chiarchaic)
                    #smallest chi squared is the style of char
                    if stylename == 'archaic' and minchi == chiarchaic : cor+=1
                    if stylename == 'hasmonean' and minchi == chihasmonean: cor+=1
                    if stylename =='herodian' and minchi == chiherodian: cor+=1
                    
                    # if minchi == chiarchaic: predicted ='archaic'
                    # if minchi == chihasmonean: predicted = 'hasmonean'
                    # if minchi ==chiherodian: predicted = 'herodian'
                    # if stylename != predicted:
                    #     print(' Wrong classification')
                    #     print('true:',stylename)
                    #     print('label:',label)
                    #     print('predicted:',predicted)

    count = 0
    # get total number of instances
    for stylename,styledataset in dataset.items():
        for label,characterset in styledataset.items():
            for image in characterset:
                count+=1
    
    print('---',count)
    print('Total accuracy:',( cor / count))
        

if __name__ == '__main__':
    # / home / jan / PycharmProjects / HandwritingRecog /
    #path for no morph
    style_base_path = 'data/characters_for_style_classification_balance_morph/'
    style_archaic_path = style_base_path + 'Archaic/'
    style_hasmonean_path = style_base_path + 'Hasmonean/'
    style_herodian_path = style_base_path + 'Herodian/'

    archaic_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Nun-final', 'He', 'Qof', 'Kaf', 'Samekh', 'Yod', 'Dalet',
                          'Waw', 'Ayin', 'Mem', 'Gimel', 'Bet', 'Shin', 'Resh', 'Alef', 'Het']

    hasmonean_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                            'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                            'Gimel',
                            'Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het']

    herodian_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                           'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                           'Gimel',
                           'Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het']

    # Retrieve img lists from each class' each character AND resize them
    new_size = (40, 40)  # change this to something which is backed up by a reason

#get image dict
    archaic_imgs = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_archaic_path, char)]
                    for char in archaic_characters}
    hasmonean_imgs ={char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_hasmonean_path, char)]
                    for char in hasmonean_characters}
    herodian_imgs = {char:
                     [cv2.resize(img, new_size) for img in get_style_char_images(style_herodian_path, char)]
                     for char in herodian_characters}

#training path for test images (for accuracy of class for individual images)
    style_base_path = 'C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/data/Style_classification/'
    style_archaic_path = style_base_path + 'Archaic/'
    style_hasmonean_path = style_base_path + 'Hasmonean/'
    style_herodian_path = style_base_path + 'Herodian/'

    archaic_nomorph = {char:
                       [cv2.resize(img, new_size) for img in get_style_char_images(style_archaic_path, char)]
                       for char in archaic_characters}
    hasmonean_nomorph = {char:
                         [cv2.resize(img, new_size) for img in get_style_char_images(style_hasmonean_path, char)]
                         for char in hasmonean_characters}
    herodian_nomorph = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_herodian_path, char)]
                    for char in herodian_characters}
        
    dataset = {'archaic':archaic_nomorph,'hasmonean':hasmonean_nomorph,'herodian':herodian_nomorph}
    get_accuracy_alldata(dataset,archaic_imgs,hasmonean_imgs,herodian_imgs)
