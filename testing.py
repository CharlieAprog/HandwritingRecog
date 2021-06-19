from Style_Classification.dim_reduction import get_style_char_images
import cv2
from Text_Segmentation.characterSegmentation import getBoundingBoxBoundaries, remove_character_artifacts
from Text_Segmentation.plotting import plotSimpleImages
from Text_Segmentation.lineSegmentation import get_binary
from skimage.filters import threshold_otsu
import numpy as np


style_base_path = 'data/characters_for_style_classification_balance_morph/'
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
new_size = (40, 40)  # change this to something which is backed up by a reason

print("working")
archaic_imgs = {char:
                [ cv2.resize(img, new_size) for img in get_style_char_images(style_archaic_path, char)]
                for char in archaic_characters}
hasmonean_imgs = {char:
                [cv2.resize(img, new_size) for img in get_style_char_images(style_hasmonean_path, char)]
                for char in hasmonean_characters}
herodian_imgs = {char:
                [cv2.resize(img, new_size) for img in get_style_char_images(style_herodian_path, char)]
                for char in herodian_characters}
all_archaic = []
for letter in archaic_imgs:
    for example in archaic_imgs[letter]:
        all_archaic.append(example)

for image in  all_archaic:
    #image = cv2.bitwise_not(image)
    image = get_binary(image).astype(np.uint8)
    new = remove_character_artifacts(image, min_cluster= 5000)   
    plotSimpleImages([image,new], title='somethin')