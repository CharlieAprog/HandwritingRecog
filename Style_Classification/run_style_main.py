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
                          'Waw', 'Ayin', 'Mem', 'Gimel', 'Bet', 'Shin', 'Resh', 'Alef', 'Het', 'Global']

    hasmonean_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                            'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                            'Gimel','Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het', 'Global']

    herodian_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                           'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                           'Gimel','Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het', 'Global']

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
    style_base_path = '/home/jan/PycharmProjects/HandwritingRecog/data/Style_train_test/'
    style_archaic_path = style_base_path + 'style_train_test_arch/train/'
    style_hasmonean_path = style_base_path + 'style_train_test_has/train/'
    style_herodian_path = style_base_path + 'style_train_test_hero/train/'

    style_archaic_path_test = style_base_path + 'style_train_test_arch/val/'
    style_hasmonean_path_test = style_base_path + 'style_train_test_has/val/'
    style_herodian_path_test = style_base_path + 'style_train_test_hero/val/'

    new_size_x, new_size_y = 40, 40  # change this to something which is backed up by a reason
    archaic_train = {char:
                       [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images(style_archaic_path, char)]
                       for char in archaic_characters}
    hasmonean_train = {char:
                         [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images(style_hasmonean_path, char)]
                         for char in hasmonean_characters}
    herodian_train = {char:
                    [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images(style_herodian_path, char)]
                    for char in herodian_characters}

    archaic_test = {char:
                       [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images(style_archaic_path_test, char)]
                       for char in archaic_characters}
    hasmonean_test = {char:
                         [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images(style_hasmonean_path_test, char)]
                         for char in hasmonean_characters}
    herodian_test = {char:
                    [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images(style_herodian_path_test, char)]
                    for char in herodian_characters}
        
    dataset_train = {'archaic':archaic_train,'hasmonean':hasmonean_train,'herodian':herodian_train}
    dataset_test = {'archaic':archaic_test,'hasmonean':hasmonean_test,'herodian':herodian_test}
    get_accuracy_alldata(dataset_train,archaic_train,hasmonean_train,herodian_train, dataset_test)
