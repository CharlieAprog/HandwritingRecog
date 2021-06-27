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


def get_style_char_vec(characters, labels,probabilities,prob_threshold = 0.5,global_vec = False, show_hinge_points=False):
    # main pipeline function to get char
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
    new_size_x, new_size_y = 40, 40  # change this to something which is backed up by a reason

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
    print('old:')
    print(len(characters))
    print(labels)
    index = np.where(probabilities <= prob_threshold)
    print(index)
    characters = np.delete(characters,index)
    labels = np.delete(labels,index)
    probabilities = np.delete(probabilities,index)
    print('now')
    print(len(characters))
    print(labels)

    print('characters thresholded out:',sum(index))

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
            # print(archaic_pdfs['Global'])

    else:
        print('getting pdfs')

        archaic_pdfs = get_hinge_pdf(idx2name[27],archaic_imgs)
        hasmonean_pdfs = get_hinge_pdf(idx2name[27],hasmonean_imgs)
        herodian_pdfs= get_hinge_pdf(idx2name[27],herodian_imgs)
        # np.save("archaic_pdfs", archaic_pdfs)
        # np.save("hasmonean_pdfs", hasmonean_pdfs)
        # np.save("herodian_pdfs", herodian_pdfs)
        #
        # archaic_pdfs = np.load("/home/jan/PycharmProjects/HandwritingRecog/data/Style_classification_pdfs/archaic_pdfs.npy")
        # hasmonean_pdfs = np.load("/home/jan/PycharmProjects/HandwritingRecog/data/Style_classification_pdfs/hasmonean_pdfs.npy")
        # herodian_pdfs = np.load("/home/jan/PycharmProjects/HandwritingRecog/data/style_classification_pdfs/herodian_pdfs.npy")

        for image, label in zip(characters, labels):
            feature_vector = get_char_vector(image)

            # chi_archaic,p = stats.chi2_contingency(feature_vector+archaic_pdfs)
            # chi_hasmonean,_ = stats.chi2(feature_vector, hasmonean_pdfs)
            # chi_herodian,_ = stats.chi2(feature_vector, herodian_pdfs)
            # print(p)
            chi_archaic = get_chisquared(feature_vector, archaic_pdfs)
            chi_hasmonean = get_chisquared(feature_vector, hasmonean_pdfs)
            chi_herodian = get_chisquared(feature_vector, herodian_pdfs)

            minchi = min(chi_hasmonean, chi_herodian, chi_archaic)
            if minchi == chi_archaic: predicted = 'Archaic'
            if minchi == chi_hasmonean: predicted = 'Hasmonean'
            if minchi == chi_herodian: predicted = 'Herodian'

            style_char_vec.append(predicted)
            chi_squared_vec.append(minchi)

    return style_char_vec, chi_squared_vec


def get_dominant_style(style_vec, chisquared_vec,n_neighbors = 10 ):
    style_vec = [sorting for _, sorting in sorted(zip(chisquared_vec, style_vec))]
    return Counter(style_vec[:n_neighbors])


def get_accuracy_alldata(dataset, archaic_imgs, hasmonean_imgs, herodian_imgs, dataset_test):
    print("Getting accuracy on the whole char set")
    archaic_pdfs = {}
    hasmonean_pdfs = {}
    herodian_pdfs = {}
    idx2name = {0: 'Alef', 1: 'Ayin', 2: 'Bet', 3: 'Dalet', 4: 'Gimel', 5: 'He',
                6: 'Het', 7: 'Kaf', 8: 'Kaf-final', 9: 'Lamed', 10: 'Mem',
                11: 'Mem-medial', 12: 'Nun-final', 13: 'Nun-medial', 14: 'Pe',
                15: 'Pe-final', 16: 'Qof', 17: 'Resh', 18: 'Samekh', 19: 'Shin',
                20: 'Taw', 21: 'Tet', 22: 'Tsadi-final', 23: 'Tsadi-medial',
                24: 'Waw', 25: 'Yod', 26: 'Zayin', 27: 'Global'}
    img_in_train =0
    # calculate all codebook vectors and store them in dict
    for stylename, styledataset in dataset.items():
        for label, characterset in styledataset.items():
            for image in characterset:
                img_in_train += 1
            # arhaic dataset doesnt have theselabels: we only label these as hasmonena & herodian
            if (label == 'Tet' or label == 'Tsadi-final' or label == 'Nun-medial' or label == 'Mem-medial'
                    or label == 'Pe-final' or label == 'Zayin' or label == 'Tsadi-medial'):
                if idx2name[27] not in archaic_pdfs:
                    archaic_pdfs[idx2name[27]] = get_hinge_pdf(idx2name[27], archaic_imgs)
                if idx2name[27] not in hasmonean_pdfs:
                    hasmonean_pdfs[idx2name[27]] = get_hinge_pdf(idx2name[27], hasmonean_imgs)
                if idx2name[27] not in herodian_pdfs:
                    herodian_pdfs[idx2name[27]] = get_hinge_pdf(idx2name[27], herodian_imgs)
            else:
                # get hinge pdfs
                if label not in archaic_pdfs:
                    archaic_pdfs[label] = get_hinge_pdf(label, archaic_imgs)
                if label not in hasmonean_pdfs:
                    hasmonean_pdfs[label] = get_hinge_pdf(label, hasmonean_imgs)
                if label not in herodian_pdfs:
                    herodian_pdfs[label] = get_hinge_pdf(label, herodian_imgs)
    print(archaic_pdfs['Global'])
    print(hasmonean_pdfs['Global'])
    print("Images used to construct codebook vector: ", img_in_train)
    cor = 0
    total = 0
    cor_global = 0
    total_global = 0
    ignored = 0
    cor_arch, cor_hero, cor_has = 0, 0 , 0
    wrong_arch, wrong_has, wrong_hero = 0, 0, 0
    for stylename, styledataset_test in dataset_test.items():
        for label, characterset in styledataset_test.items():
            for image in characterset:
                if (label == 'Tet' or label == 'Tsadi-final' or label == 'Nun-medial' or label == 'Mem-medial'
                        or label == 'Pe-final' or label == 'Zayin' or label == 'Tsadi-medial'):
                    # calculate vector for char and chisquared distance
                    feature_vector = get_char_vector(image, False)
                    if feature_vector != 0:
                        chiarchaic = get_chisquared(feature_vector, archaic_pdfs['Global'])
                        chihasmonean = get_chisquared(feature_vector, hasmonean_pdfs['Global'])
                        chiherodian = get_chisquared(feature_vector, herodian_pdfs['Global'])
                        minchi = min(chihasmonean, chiherodian, chiarchaic)
                        # smallest chi squared is the style of char
                        if stylename == 'archaic':
                            if minchi == chiarchaic:
                                cor_global += 1
                                cor_arch += 1
                            else:
                                wrong_arch += 1
                        if stylename == 'hasmonean':
                            if minchi == chihasmonean:
                                cor_global += 1
                                cor_has += 1
                            else:
                                wrong_has += 1
                        if stylename == 'herodian':
                            if minchi == chiherodian:
                                cor_global += 1
                                cor_hero += 1
                            else:
                                wrong_hero += 1
                        total_global += 1
                    else:
                        ignored += 1
                else:
                    feature_vector = get_char_vector(image, False)
                    if feature_vector != 0:
                        chiarchaic = get_chisquared(feature_vector, archaic_pdfs[label])
                        chihasmonean = get_chisquared(feature_vector, hasmonean_pdfs[label])
                        chiherodian = get_chisquared(feature_vector, herodian_pdfs[label])
                        minchi = min(chihasmonean, chiherodian, chiarchaic)
                        # smallest chi squared is the style of char
                        if stylename == 'archaic':
                            if minchi == chiarchaic:
                                cor += 1
                                cor_arch += 1
                            else:
                                wrong_arch += 1
                        if stylename == 'hasmonean':
                            if minchi == chihasmonean:
                                cor += 1
                                cor_has += 1
                            else:
                                wrong_has += 1
                        if stylename == 'herodian':
                            if minchi == chiherodian:
                                cor += 1
                                cor_hero += 1
                            else:
                                wrong_hero += 1
                        total += 1
                    else:
                        ignored += 1



                # if minchi == chihasmonean: predicted = 'hasmonean'
                # if minchi ==chiherodian: predicted = 'herodian'
                # if stylename != predicted :
                #     print(' wrong classification')
                #     print('true:',stylename)
                #     print('label:',label)
                #     print('predicted:',predicted)

    print('Total characters', total+total_global)
    print("Ignored: ", ignored)
    print('Label specific accuracy:', (cor / total))
    print('Global accuracy:', (cor_global / total_global))
    print("Archaic correct vs wrong", cor_arch, wrong_arch)
    print("Hasmonean correct vs wrong", cor_has, wrong_has)
    print("Herodian correct vs wrong", cor_hero, wrong_hero)