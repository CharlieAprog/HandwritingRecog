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
from Style_Classification.hinge_feature_calc import *

PI = 3.14159265359


def get_style_char_vec(characters, labels,global_vec = False, show_hinge_points=0):
    # main pipeline function to get char
    style_char_vec = []
    chi_squared_vec = []
    style_base_path = '/home/jan/PycharmProjects/HandwritingRecog/data/characters_for_style_classification_balance_morph/'

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
    new_size_x, new_size_y = 40, 40  # change this to something which is backed up by a reason

    # get image dict
    archaic_imgs = {char:
                        [resize_pad(img, new_size_x, new_size_y) for img in
                         get_style_char_images(style_base_path + 'Archaic/', char)]
                    for char in archaic_characters}
    hasmonean_imgs = {char:
                          [resize_pad(img, new_size_x, new_size_y) for img in
                           get_style_char_images(style_base_path + 'Hasmonean/', char)]
                      for char in hasmonean_characters}
    herodian_imgs = {char:
                         [resize_pad(img, new_size_x, new_size_y) for img in
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


    print(len(archaic_imgs['Global']))
    print(len(hasmonean_imgs['Global']))
    print(len(herodian_imgs['Global']))

    if global_vec == False:

        for image, label in zip(characters, labels):
            if (label == 21 or label == 22 or label == 13 or label == 11
                    or label == 15 or label == 26 or label == 23):
                if idx2name[27] not in archaic_pdfs:
                    archaic_pdfs[idx2name[27]] = get_hinge_pdf(idx2name[27], archaic_imgs)
                if idx2name[27] not in hasmonean_pdfs:
                    hasmonean_pdfs[idx2name[27]] = get_hinge_pdf(idx2name[27], hasmonean_imgs)
                if idx2name[27] not in herodian_pdfs:
                    herodian_pdfs[idx2name[27]] = get_hinge_pdf(idx2name[27], herodian_imgs)

                    # get feature vector of given char and get chisquared for each pdf
                feature_vector = get_char_vector(image)
                chihasmonean = get_chisquared(feature_vector, hasmonean_pdfs[idx2name[27]])
                chiherodian = get_chisquared(feature_vector, herodian_pdfs[idx2name[27]])
                chiarchaic = get_chisquared(feature_vector, archaic_pdfs[idx2name[27]])
                minchi = min(chihasmonean, chiherodian, chiarchaic)

                if minchi == chihasmonean: predicted = 'Hasmonean'
                if minchi == chiherodian: predicted = 'Herodian'
                if minchi == chiarchaic: predicted = 'Archaic'
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
                feature_vector = get_char_vector(image, cnt=show_hinge_points)
                chiarchaic = get_chisquared(feature_vector, archaic_pdfs[idx2name[label]])
                chihasmonean = get_chisquared(feature_vector, hasmonean_pdfs[idx2name[label]])
                chiherodian = get_chisquared(feature_vector, herodian_pdfs[idx2name[label]])

                # smallest chi squared is the style of char
                minchi = min(chihasmonean, chiherodian, chiarchaic)
                if minchi == chiarchaic: predicted = 'Archaic'
                if minchi == chihasmonean: predicted = 'Hasmonean'
                if minchi == chiherodian: predicted = 'Herodian'
                style_char_vec.append(predicted)
                chi_squared_vec.append(minchi)
            # print(archaic_pdfs['Global'])

    else:
        archaic_pdfs = get_hinge_pdf(idx2name[27],archaic_imgs)
        hasmonean_pdfs = get_hinge_pdf(idx2name[27],hasmonean_imgs)
        herodian_pdfs= get_hinge_pdf(idx2name[27],herodian_imgs)
        print('global_pdfs!!!')
        print(archaic_pdfs)
        print(len(archaic_pdfs))
        print('--------------------------------------------------------')
        print(hasmonean_pdfs)
        print(len(hasmonean_pdfs))
        print('--------------------------------------------------------')
        print(herodian_pdfs)
        print(len(herodian_pdfs))
        
        for image, label in zip(characters, labels):
            feature_vector = get_char_vector(image, cnt=0)
            chiarchaic = get_chisquared(feature_vector, archaic_pdfs)
            chihasmonean = get_chisquared(feature_vector, hasmonean_pdfs)
            chiherodian = get_chisquared(feature_vector, herodian_pdfs)
            minchi = min(chihasmonean, chiherodian, chiarchaic)
            if minchi == chiarchaic: predicted = 'Archaic'
            if minchi == chihasmonean: predicted = 'Hasmonean'
            if minchi == chiherodian: predicted = 'Herodian'
            style_char_vec.append(predicted)
            chi_squared_vec.append(minchi)

    return style_char_vec, chi_squared_vec


def get_dominant_style(style_vec, chisquared_vec):
    style_vec = [sorting for _, sorting in sorted(zip(chisquared_vec, style_vec))]
    return Counter(style_vec)


def get_accuracy_alldata(dataset, archaic_imgs, hasmonean_imgs, herodian_imgs):
    ##Get accuracy: for finetuning and testing purposes
    cor = 0
    for stylename, styledataset in dataset.items():
        for label, characterset in styledataset.items():
            # arhaic dataset doesnt have theselabels: we only label these as hasmonena & herodian
            if (label == 'Tet' or label == 'Tsadi-final' or label == 'Nun-medial' or label == 'Mem-medial'
                    or label == 'Pe-final' or label == 'Zayin' or label == 'Tsadi-medial'):
                hasmonean_pdf = get_hinge_pdf(label, hasmonean_imgs)
                herodian_pdf = get_hinge_pdf(label, herodian_imgs)
                for image in characterset:
                    # get feature vector of given char and get chisquared for each pdf
                    feature_vector = get_char_vector(image)
                    chihasmonean = get_chisquared(feature_vector, hasmonean_pdf)
                    chiherodian = get_chisquared(feature_vector, herodian_pdf)
                    minchi = min(chihasmonean, chiherodian)

                    # classify and debug etc
                    if stylename == 'hasmonean' and minchi == chihasmonean:
                        cor += 1
                    if stylename == 'herodian' and minchi == chiherodian:
                        cor += 1

                    # if minchi == chihasmonean: predicted = 'hasmonean'
                    # if minchi ==chiherodian: predicted = 'herodian'
                    # if stylename != predicted :
                    #     print(' wrong classification')
                    #     print('true:',stylename)
                    #     print('label:',label)
                    #     print('predicted:',predicted)
            else:
                # get hinge pdfs
                archaic_pdf = get_hinge_pdf(label, archaic_imgs)
                hasmonean_pdf = get_hinge_pdf(label, hasmonean_imgs)
                herodian_pdf = get_hinge_pdf(label, herodian_imgs)

                for image in characterset:
                    # calculate vector for char and chisquared distance
                    feature_vector = get_char_vector(image)
                    chiarchaic = get_chisquared(feature_vector, archaic_pdf)
                    chihasmonean = get_chisquared(feature_vector, hasmonean_pdf)
                    chiherodian = get_chisquared(feature_vector, herodian_pdf)

                    minchi = min(chihasmonean, chiherodian, chiarchaic)
                    # smallest chi squared is the style of char
                    if stylename == 'archaic' and minchi == chiarchaic: cor += 1
                    if stylename == 'hasmonean' and minchi == chihasmonean: cor += 1
                    if stylename == 'herodian' and minchi == chiherodian: cor += 1

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
    for stylename, styledataset in dataset.items():
        for label, characterset in styledataset.items():
            for image in characterset:
                count += 1

    print('---', count)
    print('Total accuracy:', (cor / count))