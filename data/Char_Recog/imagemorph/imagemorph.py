import sys
import PIL 
from ctypes import *
from pathlib import Path

import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import random

class Pixel(Structure):
    """ This class mimics the Pixel structure defined in imagemorph.c. """
    _fields_ = [('r', c_int),
                ('g', c_int),
                ('b', c_int)]


def imagemorph(img, amp, sigma, h, w):
    """ 
    Apply random elastic morphing to an image. 

    Args:
        img: BGR image in the form of a numpy array of shape (h, w, 3). 
        amp: average amplitude of the displacement field (average pixel displacement)
        sigma: standard deviation of the Gaussian smoothing kernel 
        h: height of the image
        w: width of the image
    """
    assert img.shape == (h, w, 3), f"img should have shape (h, w, 3), not {img.shape}"

    # load C library
    try:
        cwd = Path(__file__).resolve().parent  # location of this module
        libfile = list(cwd.rglob('imagemorph*.so'))[0]
    except IndexError:
        print("Error: imagemorph library could not be found. Make sure to "
              "first compile the C library using `python setup.py build`.")
        sys.exit()

    c_lib = CDLL(libfile)

    # load the imagemorph function from the library
    imagemorph = c_lib.imagemorph
    imagemorph.restype = POINTER(POINTER(Pixel))
    imagemorph.argtypes = [POINTER(POINTER(Pixel)), c_int, c_int, c_double, c_double]

    # convert parameters to C compatible data types
    img_c = (h * POINTER(Pixel))()
    for i in range(h):
        row = (w * Pixel)()
        for j in range(w):
            b, g, r = img[i, j]
            row[j] = Pixel(r, g, b)
        img_c[i] = cast(row, POINTER(Pixel))
    img_c = cast(img_c, POINTER(POINTER(Pixel)))
    amp_c, sigma_c = c_double(amp), c_double(sigma)
    h_c, w_c = c_int(h), c_int(w)
    
    # apply the imagemorph function to the image
    img_c = imagemorph(img_c, h_c, w_c, amp_c, sigma_c)

    # convert the result to a numpy array
    res = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            px = img_c[i][j]
            res[i, j] = [px.b, px.g, px.r]
    return res


# if __name__ == '__main__':
#     amp, sigma = 0.5, 1.5
#     img_name = "img/sample-input.png"

#     # load image
#     img = cv.imread(img_name)
#     h, w, _ = img.shape

#     # apply imagemorph
#     res = imagemorph(img, amp, sigma, h, w)

#     # write result to disk
#     cv.imwrite('img/out.png', res)

# if line with morph is not commented this function take a path to folders (with all the chars)
# then it first gets the distribution of classes and afterwards performs augmentation so that
# all classes are relatively balanced (max instances per class = 2* min instances per class)
# these new images are saved in a new folder that has to be created before
def imagemorph_folder_balance(folder_path, double_current_data=False, even_data=False):
    image_names = glob.glob(folder_path)
    label_distribution = {}
    folders = image_names[0].split('/')
    prev_label = folders[8]
    print(prev_label)
    label_cnt = 0
    for img_name in image_names:
        folders = img_name.split('/')
        label = folders[8]
        if label == prev_label:
            label_cnt += 1
        else:
            label_distribution[prev_label] = label_cnt + 1
            label_cnt = 0
        prev_label = label
    label_distribution[label] = label_cnt
    print(label_distribution)
    cnt = 0
    # if data should be saved in different folder
    # path = '/home/jan/PycharmProjects/HandwritingRecog/data/Char_Recog/binarized_hdd_40x40'
    path = '/home/jan/PycharmProjects/HandwritingRecog/data/Char_Recog/grayscale_hhd_224x224'
    if double_current_data:
        amp = random.uniform(1.1, 1.6)
        sigma = random.uniform(2.5, 4.0)
        for img_name in image_names:
            folders = img_name.split('/')
            label = folders[8]
            img = cv.imread(img_name)
            h, w, _ = img.shape
            res = imagemorph(img, amp, sigma, h, w)
            out_name = label + '/' + 'double' + str(cnt) + '.png'
            x = cv.imwrite(os.path.join(path, out_name), res)
            print(x)
            cnt += 1
    if even_data:
        max_label_cnt = max(label_distribution.values())
        for label in label_distribution:
            if label_distribution[label] < (max_label_cnt/2):
                # how many times should we morph
                morph_times = int((max_label_cnt/2) / label_distribution[label])
                #print('morphing', morph_times)
                morph(folder_path, label, morph_times)

def morph(folder_path, label_to_morph, morph_times, new_folder=False):
    image_names = glob.glob(folder_path)
    cnt = 0
    # path to where the morphed images are saved, each char will get its own folder
    path = '/home/jan/PycharmProjects/HandwritingRecog/data/Char_Recog/binarized_hhd_40x40'
    #os.chdir(path)
    folders = image_names[0].split('/')
    prev_label = folders[8]
    for i in range(0, morph_times):
        if i == 0 and new_folder:
            only_one_directory = True
        else:
            only_one_directory = False
        # use randomly choosen amp and sigma
        amp = random.uniform(1.1, 1.6)
        sigma = random.uniform(2.5, 4.0)
        for img_name in image_names:
            folders = img_name.split('/')
            label = folders[8]
            if label == label_to_morph:
                if only_one_directory and label != prev_label:
                    os.mkdir(os.path.join(path, label))
                    prev_label = label
                img = cv.imread(img_name)
                h, w, _ = img.shape
                res = imagemorph(img, amp, sigma, h, w)
                out_name = label + '/' + str(cnt) + '.png'
                x = cv.imwrite(os.path.join(path, out_name), res)
                print(x)
                cnt += 1


if __name__ == '__main__':
    imagemorph_folder_balance('/home/jan/PycharmProjects/HandwritingRecog/data/Char_Recog/grayscale_hhd_224x224/*/*.png', double_current_data=True)