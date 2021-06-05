import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from typing import Union

def get_style_char_images(style_path: str, character: str):
    """ Returns a list of numpy arrays, where each arrays corresponds to an image from a certain character of a
    certain class """
    img_name_list = glob.glob(f"{style_path}{character}/*.jpg")
    img_list = [cv2.imread(img, 0) for img in img_name_list]
    return img_list


style_base_path = '../data/characters_for_style_classification/'
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
new_size = (32, 32)
archaic_imgs = {char:
                [cv2.resize(img, new_size) for img in get_style_char_images(style_archaic_path, char)]
                for char in archaic_characters}
hasmonean_imgs = {char:
                  [cv2.resize(img, new_size) for img in get_style_char_images(style_hasmonean_path, char)]
                  for char in hasmonean_characters}
herodian_imgs = {char:
                 [cv2.resize(img, new_size) for img in get_style_char_images(style_herodian_path, char)]
                 for char in herodian_characters}

pca = PCA(2)
pca_archaic_imgs = {char:
                [pca.fit_transform()]
                for char in archaic_characters}
    pca.fit_transform()


digits = load_digits()
data = digits.data
img = data[0, :].reshape(8, 8)


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 6))
color_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s=15,
            cmap=color_map, c=digits.target)
plt.colorbar()
plt.xlabel('PC-1'), plt.ylabel('PC-2')
plt.show()
