
from feature_detection import *

def get_style_char_images_train_test(style_path: str, character: str):
    """ Returns a list of numpy arrays, where each arrays corresponds to an image from a certain character of a
    certain class """
    if character == 'Global':
        list_for_glob1 = style_path + '*/*.jpg'
    else:
        list_for_glob1 = style_path + character + '/*.jpg'
    if character == 'Global':
        list_for_glob2 = style_path + '*/*.png'
    else:
        list_for_glob2 = style_path + character + '/*.png'
    img_name_list1 = glob.glob(list_for_glob1)
    img_name_list2 = glob.glob(list_for_glob2)
    img_name_list = img_name_list1 + img_name_list2
    img_list = [cv2.imread(img, 0) for img in img_name_list]
    assert len(img_list) > 0, "Trying to read image files while being in a wrong folder."
    return img_list

if __name__ == '__main__':
    archaic_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Nun-final', 'He', 'Qof', 'Kaf', 'Samekh', 'Yod', 'Dalet',
                          'Waw', 'Ayin', 'Mem', 'Gimel', 'Bet', 'Shin', 'Resh', 'Alef', 'Het', 'Global']

    hasmonean_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                            'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                            'Gimel','Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het', 'Global']

    herodian_characters = ['Taw', 'Pe', 'Kaf-final', 'Lamed', 'Tet', 'Nun-final', 'Tsadi-final', 'He', 'Qof', 'Kaf',
                           'Samekh', 'Yod', 'Dalet', 'Waw', 'Ayin', 'Mem-medial', 'Nun-medial', 'Mem', 'Pe-final',
                           'Gimel','Bet', 'Shin', 'Resh', 'Zayin', 'Alef', 'Tsadi-medial', 'Het', 'Global']


#training path for test images (for accuracy of class for individual images)
    #training path
    style_base_path = 'data/train_test_morph/'
    style_archaic_path = style_base_path + 'style_train_test_arch_morph/train/'
    style_hasmonean_path = style_base_path + 'style_train_test_has_morph/train/'
    style_herodian_path = style_base_path + 'style_train_test_hero_morph/train/'

    #testin path
    style_archaic_path_test = style_base_path + 'style_train_test_arch_morph/val/'
    style_hasmonean_path_test = style_base_path + 'style_train_test_has_morph/val/'
    style_herodian_path_test = style_base_path + 'style_train_test_hero_morph/val/'

    
    
    new_size_x, new_size_y = 40, 40  # change this to something which is backed up by a reason
    archaic_train = {char:
                       [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images_train_test(style_archaic_path, char)]
                       for char in archaic_characters}

    hasmonean_train = {char:
                         [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images_train_test(style_hasmonean_path, char)]
                         for char in hasmonean_characters}

    herodian_train = {char:
                    [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images_train_test(style_herodian_path, char)]
                    for char in herodian_characters}

    archaic_test = {char:
                       [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images_train_test(style_archaic_path_test, char)]
                       for char in archaic_characters}

    hasmonean_test = {char:
                         [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images_train_test(style_hasmonean_path_test, char)]
                         for char in hasmonean_characters}

    herodian_test = {char:
                    [resize_pad(img, new_size_x, new_size_y, 255) for img in get_style_char_images_train_test(style_herodian_path_test, char)]
                    for char in herodian_characters}
        
    dataset_train = {'archaic':archaic_train,'hasmonean':hasmonean_train,'herodian':herodian_train}
    dataset_test = {'archaic':archaic_test,'hasmonean':hasmonean_test,'herodian':herodian_test}
    get_accuracy_alldata(dataset_train,archaic_train,hasmonean_train,herodian_train, dataset_test)
