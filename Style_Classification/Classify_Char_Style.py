import math
import cv2
from skimage import feature
from Style_Classification.hinge_utils import *
from Text_Segmentation.segmentation_to_recog import resize_pad
# from segmentation_to_recog import resize_pad
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import stats
from sklearn.decomposition import PCA
import glob
from collections import Counter
from Style_Classification.Calculate_Hinge_Features import *

PI = 3.14159265359


'''
Main function to classify a given list of character images and their labels.
All needed codebook vectors are calculated and character images are classfied based
on Chi-Squared distance.
'''
def get_style_char_vec(characters, labels,probabilities,prob_threshold = 0.0,global_vec = False, show_hinge_points=False):
    # main pipeline function to get char style for a given vector
    style_char_vec = []
    chi_squared_vec = []
    style_base_path = 'data/Style_classification/'

    archaic_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Nun-final', 'He', 'Qof', 'Kaf', 'Samekh', 'Yod', 'Dalet',
                          'Waw', 'Ayin', 'Mem', 'Gimel', 'Bet', 'Shin', 'Resh', 'Alef', 'Het', 'Global']

    hasmonean_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                            'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                            'Gimel',
                            'Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het', 'Global']

    herodian_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                           'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                           'Gimel',
                           'Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het', 'Global']

    # Retrieve img lists from each class' each character AND resize them
    new_size_x, new_size_y = 40, 40  #in accordance to CNN transformation of data

    # get image dict
    archaic_imgs = {char:
                        [resize_pad(img, new_size_x, new_size_y, 255) for img in
                         get_style_char_images(style_base_path + 'Archaic/', char)]
                    for char in archaic_characters}

    hasmonean_imgs = {char:
                          [resize_pad(img, new_size_x, new_size_y, 255) for img in
                           get_style_char_images(style_base_path + 'Hasmonean/', char)]
                      for char in hasmonean_characters}

    herodian_imgs = {char:
                         [resize_pad(img, new_size_x, new_size_y, 255) for img in
                          get_style_char_images(style_base_path + 'Herodian/', char)]
                     for char in herodian_characters}

    idx2name = {0: 'Alef', 1: 'Ayin', 2: 'Bet', 3: 'Dalet', 4: 'Gimel', 5: 'He',
                6: 'Het', 7: 'Kaf', 8: 'Kaf-final', 9: 'Lamed', 10: 'Mem',
                11: 'Mem-medial', 12: 'Nun-final', 13: 'Nun-medial', 14: 'Pe',
                15: 'Pe-final', 16: 'Qof', 17: 'Resh', 18: 'Samekh', 19: 'Shin',
                20: 'Taw', 21: 'Tet', 22: 'Tsadi-final', 23: 'Tsadi-medial',
                24: 'Waw', 25: 'Yod', 26: 'Zayin', 27: 'Global'}

    archaic_pdfs = {}
    hasmonean_pdfs = {}
    herodian_pdfs = {}

    #style classification for chars with a high prob value;exclude chars<=thresh
    print('#Segmented characters from pipeline')
    print(len(characters))

    index = np.where(probabilities <= prob_threshold)
    characters = np.delete(characters,index)
    labels = np.delete(labels,index)
    probabilities = np.delete(probabilities,index)

    print('#Number of charactersafter removal by probability thresholding')
    print(len(characters))

    # calculate Character specific hinge pdfs per style
    if global_vec == False:

        for image, label in zip(characters, labels):
            #some labels do not exist in archaic. Account for that
            if (label == 21 or label == 22 or label == 13 or label == 11
                    or label == 15 or label == 26 or label == 23):

                #Get Hinge pdfs
                if idx2name[27] not in archaic_pdfs:
                    archaic_pdfs[idx2name[27]] = get_hinge_pdf(idx2name[27], archaic_imgs)
                if idx2name[27] not in hasmonean_pdfs:
                    hasmonean_pdfs[idx2name[27]] = get_hinge_pdf(idx2name[27], hasmonean_imgs)
                if idx2name[27] not in herodian_pdfs:
                    herodian_pdfs[idx2name[27]] = get_hinge_pdf(idx2name[27], herodian_imgs)

                # get feature vector of given char and get chisquared for each pdf
                feature_vector = get_char_vector(image)
                if feature_vector != 0:
                    chi_hasmonean = get_chisquared(feature_vector, hasmonean_pdfs[idx2name[27]])
                    chi_herodian = get_chisquared(feature_vector, herodian_pdfs[idx2name[27]])
                    chi_archaic = get_chisquared(feature_vector, archaic_pdfs[idx2name[27]])
                    minchi = min(chi_hasmonean, chi_herodian, chi_archaic)

                    if minchi == chi_hasmonean: predicted = 'Hasmonean'
                    if minchi == chi_herodian: predicted = 'Herodian'
                    if minchi == chi_archaic: predicted = 'Archaic'

                    style_char_vec.append(predicted)
                    chi_squared_vec.append(minchi)

            else:
                # get hinge pdfs
                if idx2name[label] not in archaic_pdfs:
                    archaic_pdfs[idx2name[label]] = get_hinge_pdf(idx2name[label], archaic_imgs)
                if idx2name[label] not in hasmonean_pdfs:
                    hasmonean_pdfs[idx2name[label]] = get_hinge_pdf(idx2name[label], hasmonean_imgs)
                if idx2name[label] not in herodian_pdfs:
                    herodian_pdfs[idx2name[label]] = get_hinge_pdf(idx2name[label], herodian_imgs)
                
                # calculate vector for char and chisquared distance
                feature_vector = get_char_vector(image)
                if feature_vector != 0:
                    chi_archaic = get_chisquared(feature_vector, archaic_pdfs[idx2name[label]])
                    chi_hasmonean = get_chisquared(feature_vector, hasmonean_pdfs[idx2name[label]])
                    chi_herodian = get_chisquared(feature_vector, herodian_pdfs[idx2name[label]])

                    # smallest chi squared is the style of char
                    minchi = min(chi_hasmonean, chi_herodian, chi_archaic)
                    if minchi == chi_archaic: predicted = 'Archaic'
                    if minchi == chi_hasmonean: predicted = 'Hasmonean'
                    if minchi == chi_herodian: predicted = 'Herodian'
                    style_char_vec.append(predicted)
                    chi_squared_vec.append(minchi)

    else:
        print('Getting Style codebook vectors')

        # create the three codebook vectors for every style
        archaic_pdfs = get_hinge_pdf(idx2name[27],archaic_imgs)
        hasmonean_pdfs = get_hinge_pdf(idx2name[27],hasmonean_imgs)
        herodian_pdfs= get_hinge_pdf(idx2name[27],herodian_imgs)

        for image, label in zip(characters, labels):
            feature_vector = get_char_vector(image)
            if feature_vector != 0:
                chi_archaic = get_chisquared(feature_vector, archaic_pdfs)
                chi_hasmonean = get_chisquared(feature_vector, hasmonean_pdfs)
                chi_herodian = get_chisquared(feature_vector, herodian_pdfs)

                minchi = min(chi_hasmonean, chi_herodian, chi_archaic)
                if minchi == chi_archaic: predicted = 'Archaic'
                if minchi == chi_hasmonean: predicted = 'Hasmonean'
                if minchi == chi_herodian: predicted = 'Herodian'

                style_char_vec.append(predicted)
                chi_squared_vec.append(minchi)

    #return vector with char styles and vector w/ chi2 values
    return style_char_vec, chi_squared_vec

##Get dominant style of image: n nearest neigbors classification
def get_dominant_style(style_vec, chisquared_vec,n_neighbors = 10 ):
    style_vec = [sorting for _, sorting in sorted(zip(chisquared_vec, style_vec))]
    return Counter(style_vec[:n_neighbors])


