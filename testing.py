from Style_Classification.dim_reduction import get_style_char_images
import cv2
from Text_Segmentation.characterSegmentation import getBoundingBoxBoundaries, remove_character_artifacts
from Text_Segmentation.plotting import plotSimpleImages
from Text_Segmentation.lineSegmentation import get_binary
from skimage.filters import threshold_otsu
import numpy as np
from Text_Segmentation.textSegmentation import trim_360


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
    print(letter)
    for example in archaic_imgs[letter]:
        all_archaic.append(example)

# use clean image to trim an image 
#thresh side determines the minimum number of pixels of an element that touches the image border for it to be kept
#thresh mid determines the minimum number of pixels of an element in the middle of an image for it to be kept
# will subsequently trim letter
def clean_image(image, thresh_side, thresh_mid, trim_thresh= 10):
    image = get_binary(image).astype(np.uint8)
    new = remove_character_artifacts(image, min_cluster= thresh_side, internal_min_cluster=thresh_mid)
    if new.size == 0:
        new = image
    new = trim_360(new, line_thresh=trim_thresh)
    plotSimpleImages([image,new], title='example')

for image in  all_archaic:
    clean_image(image, thresh_side = 3000, thresh_mid = 200) 


