from Style_Classification.feature_detection import *


if __name__ == '__main__':
    # / home / jan / PycharmProjects / HandwritingRecog /
    #path for no morph
#     style_base_path = '/home/jan/PycharmProjects/HandwritingRecog/data/characters_for_style_classification_balance_morph/'
#     style_archaic_path = style_base_path + 'Archaic/'
#     style_hasmonean_path = style_base_path + 'Hasmonean/'
#     style_herodian_path = style_base_path + 'Herodian/'
#
    archaic_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Nun-final', 'He', 'Qof', 'Kaf', 'Samekh', 'Yod', 'Dalet',
                          'Waw', 'Ayin', 'Mem', 'Gimel', 'Bet', 'Shin', 'Resh', 'Alef', 'Het']

    hasmonean_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                            'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                            'Gimel','Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het']

    herodian_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                           'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                           'Gimel','Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het']

#     # Retrieve img lists from each class' each character AND resize them

#
# #get image dict
#     archaic_imgs = {char:
#                     [resize_pad(img, new_size_x, new_size_y) for img in get_style_char_images(style_archaic_path, char)]
#                     for char in archaic_characters}
#     hasmonean_imgs ={char:
#                     [resize_pad(img, new_size_x, new_size_y) for img in get_style_char_images(style_hasmonean_path, char)]
#                     for char in hasmonean_characters}
#     herodian_imgs = {char:
#                      [resize_pad(img, new_size_x, new_size_y) for img in get_style_char_images(style_herodian_path, char)]
#                      for char in herodian_characters}

#training path for test images (for accuracy of class for individual images)
    style_base_path = '/home/jan/PycharmProjects/HandwritingRecog/data/characters_for_style_classification_balance_morph/'
    style_archaic_path = style_base_path + 'Archaic/'
    style_hasmonean_path = style_base_path + 'Hasmonean/'
    style_herodian_path = style_base_path + 'Herodian/'

    new_size_x, new_size_y = 40, 40  # change this to something which is backed up by a reason
    archaic_nomorph = {char:
                       [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images(style_archaic_path, char)]
                       for char in archaic_characters}
    hasmonean_nomorph = {char:
                         [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images(style_hasmonean_path, char)]
                         for char in hasmonean_characters}
    herodian_nomorph = {char:
                    [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images(style_herodian_path, char)]
                    for char in herodian_characters}
        
    dataset = {'archaic':archaic_nomorph,'hasmonean':hasmonean_nomorph,'herodian':herodian_nomorph}

    get_accuracy_alldata(dataset,archaic_nomorph,hasmonean_nomorph,herodian_nomorph)
