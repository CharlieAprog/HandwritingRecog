from Style_Classification.Classify_Char_Style import *
from Style_Classification.Calculate_Hinge_Features import *


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
    cor_arch, cor_hero, cor_has = 0, 0, 0
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

    print('Total characters', total+total_global)
    print("Ignored: ", ignored)
    print('Label specific accuracy:', (cor / total))
    print('Global accuracy:', (cor_global / total_global))
    print("Archaic correct vs wrong", cor_arch, wrong_arch)
    print("Hasmonean correct vs wrong", cor_has, wrong_has)
    print("Herodian correct vs wrong", cor_hero, wrong_hero)

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
    style_base_path = '/home/jan/PycharmProjects/HandwritingRecog/data/Style_train_test/'
    style_archaic_path = style_base_path + 'style_train_test_arch/train/'
    style_hasmonean_path = style_base_path + 'style_train_test_has/train/'
    style_herodian_path = style_base_path + 'style_train_test_hero/train/'

    #testin path
    style_archaic_path_test = style_base_path + 'style_train_test_arch/val/'
    style_hasmonean_path_test = style_base_path + 'style_train_test_has/val/'
    style_herodian_path_test = style_base_path + 'style_train_test_hero/val/'

    
    
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
