import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Union
from minisom import MiniSom


def get_style_char_images(style_path: str, character: str):
    """ Returns a list of numpy arrays, where each arrays corresponds to an image from a certain character of a
    certain class """
    list_for_glob = style_path + character + '/*.jpg'
    img_name_list = glob.glob(list_for_glob)
    img_list = [cv2.imread(img, 0) for img in img_name_list]
    assert len(img_list) > 0, "Trying to read image files while being in a wrong folder. Cd into 'Style_Classification'."
    return img_list


style_base_path = './data/characters_for_style_classification_balance_morph/'
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
# new_size = (32, 32)  # change this to something which is backed up by a reason
# archaic_imgs = {char:
#                 [cv2.resize(img, new_size).flatten() for img in get_style_char_images(style_archaic_path, char)]
#                 for char in archaic_characters}
# hasmonean_imgs = {char:
#                   [cv2.resize(img, new_size).flatten() for img in get_style_char_images(style_hasmonean_path, char)]
#                   for char in hasmonean_characters}
# herodian_imgs = {char:
#                  [cv2.resize(img, new_size).flatten() for img in get_style_char_images(style_herodian_path, char)]
#                  for char in herodian_characters}


# |-----------------------------------------------|
# |                  PCA                          |
# |-----------------------------------------------|
# pca = PCA(3)
# # into too few instances which leads to insufficient components
# # pca_archaic_imgs = {char: pca.fit_transform(archaic_imgs[char]) for char in archaic_imgs.keys()}
# pca_hasmonean_imgs = {char: pca.fit_transform(hasmonean_imgs[char]) for char in hasmonean_imgs.keys()}
# pca_herodian_imgs = {char: pca.fit_transform(herodian_imgs[char]) for char in herodian_imgs.keys()}


# hasmonean_features_taw = pca_hasmonean_imgs["Taw"]
# hasmonean_features_kaf = pca_hasmonean_imgs["Kaf"]
# hasmonean_features_alef = pca_hasmonean_imgs["Alef"]
#
# herodian_features_taw = pca_herodian_imgs["Taw"]
# herodian_features_kaf = pca_herodian_imgs["Kaf"]
# herodian_features_alef = pca_herodian_imgs["Alef"]

# 2D
# hasmonean_features_taw = pca_hasmonean_imgs["Taw"] + [3000, 3000]
# hasmonean_features_kaf = pca_hasmonean_imgs["Kaf"] + [-3000, 0]
# hasmonean_features_alef = pca_hasmonean_imgs["Alef"] + [3000, -3000]
#
# herodian_features_taw = pca_herodian_imgs["Taw"] + [3000, 3000]
# herodian_features_kaf = pca_herodian_imgs["Kaf"] + [-3000, 0]
# herodian_features_alef = pca_herodian_imgs["Alef"] + [3000, -3000]

# 3D
# hasmonean_features_taw = pca_hasmonean_imgs["Taw"] + [3000, 3000, 3000]
# hasmonean_features_kaf = pca_hasmonean_imgs["Kaf"] + [-3000, 0, 0]
# hasmonean_features_alef = pca_hasmonean_imgs["Alef"] + [3000, -3000, -3000]
#
# herodian_features_taw = pca_herodian_imgs["Taw"] + [3000, 3000, 3000]
# herodian_features_kaf = pca_herodian_imgs["Kaf"] + [-3000, 0, 0]
# herodian_features_alef = pca_herodian_imgs["Alef"] + [3000, -3000, -3000]
#
# plt.style.use('seaborn-whitegrid')
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(projection='3d')
# # ax = fig.add_subplot()
# color_map = plt.cm.get_cmap('jet', 10)

# # Green1
# ax.scatter(hasmonean_features_taw[:, 0], hasmonean_features_taw[:, 1], s=15,
#             cmap=color_map, c="#69D968")
# # Blue1
# ax.scatter(hasmonean_features_kaf[:, 0], hasmonean_features_kaf[:, 1], s=15,
#             cmap=color_map, c="#A291CE")
# # Yellow1
# ax.scatter(hasmonean_features_alef[:, 0], hasmonean_features_alef[:, 1], s=15,
#             cmap=color_map, c="#FDE551")
#
# # Green2
# ax.scatter(herodian_features_taw[:, 0], herodian_features_taw[:, 1], s=15,
#             cmap=color_map, c="#91CEC8")
# # Blue2
# ax.scatter(herodian_features_kaf[:, 0], herodian_features_kaf[:, 1], s=15,
#             cmap=color_map, c="#0550D2")
# # Yellow2
# ax.scatter(herodian_features_alef[:, 0], herodian_features_alef[:, 1], s=15,
#             cmap=color_map, c="#FDA051")
#
# # Green1
# ax.scatter(hasmonean_features_taw[:, 0], hasmonean_features_taw[:, 1], hasmonean_features_taw[:, 2], s=15,
#             cmap=color_map, c="#69D968")
# # Green2
# ax.scatter(herodian_features_taw[:, 0], herodian_features_taw[:, 1], herodian_features_taw[:, 2], s=15,
#             cmap=color_map, c="#91CEC8")
#
# # Blue1
# ax.scatter(hasmonean_features_kaf[:, 0], hasmonean_features_kaf[:, 1], hasmonean_features_kaf[:, 2], s=15,
#             cmap=color_map, c="#A291CE")
# # Blue2
# ax.scatter(herodian_features_kaf[:, 0], herodian_features_kaf[:, 1], herodian_features_kaf[:, 2], s=15,
#             cmap=color_map, c="#0550D2")
#
# # Yellow1
# ax.scatter(hasmonean_features_alef[:, 0], hasmonean_features_alef[:, 1], hasmonean_features_alef[:, 2], s=15,
#             cmap=color_map, c="#FDE551")
# # Yellow2
# ax.scatter(herodian_features_alef[:, 0], herodian_features_alef[:, 1], herodian_features_alef[:, 2], s=15,
#             cmap=color_map, c="#FDA051")
#
# ax.set_xlabel('PC-1')
# ax.set_ylabel('PC-2')
# ax.set_zlabel('PC-3') if pca.n_components == 3 else None
# plt.legend(["Taw (hasmonean)", "Taw (herodian)", "Kaf (hasmonean)", "Kaf (herodian)", "Alef (hasmonean)", "Alef (herodian)"],
#            loc="best")
# plt.show()