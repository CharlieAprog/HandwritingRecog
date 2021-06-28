import matplotlib.pyplot as plt
import numpy as np
import imageio
from skimage.feature import hog
from Style_Classification.hinge_utils import *
from numpy import asarray
from PIL import Image
from tqdm import tqdm
from Text_Segmentation.segmentation_to_recog import resize_pad
from sklearn import svm
import os
import pickle
import torchvision.transforms as transforms

def get_hog(set):
    #get histogram of oriented gradients for a data split set
    set = [hog(image, orientations=8, pixels_per_cell=(4, 4),
               cells_per_block=(1, 1)) for image in set]
    return set

# class to binarize image (so that 0 is background and 1 is a character)
class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x < self.thr).to(x.dtype)  # do not change the data type

'''
Fits the SVM to the training data, calculates accuracy on testing data and saves the trained SVM
'''
def get_acc_SVM(dataset_train, dataset_test):
    print("Running SVM for Style")
    name2idxStyle = {'archaic': 0, 'hasmonean': 1, 'herodian': 2}

    bin_transform = transforms.Compose([
        transforms.ToTensor(),
        ThresholdTransform(thr_255=200)
    ])

    train_x, train_y = [], []
    # calculate all codebook vectors and store them in dict
    for stylename, styledataset in dataset_train.items():
        for label, characterset in styledataset.items():
            for char in characterset:
                char, _ = noise_removal(char)
                char = bin_transform(char).numpy()
                train_x.append(np.asarray(char[0]))
                train_y.append(name2idxStyle[stylename])

    train_x = np.asarray(get_hog(train_x))

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(train_x, train_y)

    test_x, test_y = [], []
    for stylename, styledataset in dataset_test.items():
        for label, characterset in styledataset.items():
            for char in characterset:
                char = bin_transform(char).numpy()
                test_x.append(np.asarray(char[0]))
                test_y.append(name2idxStyle[stylename])

    test_x = np.asarray(get_hog(test_x))

    predictions = clf.predict(test_x)
    wro, cor = 0, 0
    for j in range(len(test_y)):
        if predictions[j] == test_y[j]:
            cor += 1
        else:
            wro += 1

    print('Accuracy: ', cor / (cor+wro))
    # save model to disk for later use
    filename = '../SVM_for_Style.sav'
    pickle.dump(clf, open(filename, 'wb'))


'''
Runs the SVM for each character that is nicely segmented (high probability from the recognizer)
Returns the style of that image based on simple majority voting
'''
def get_style_SVM(characters, labels,probabilities,prob_threshold = 0.8):
    # load the model from disk
    filename = 'SVM_for_Style.sav'
    clf = pickle.load(open(filename, 'rb'))

    print('#Segmented characters from pipeline')
    print(len(characters))

    index = np.where(probabilities <= prob_threshold)
    characters = np.delete(characters,index)

    print('#Number of characters after removal by probability thresholding')
    print(len(characters))

    characters_on_page = []
    for char in characters:
        resized_char = resize_pad(char, 40, 40, 0)
        characters_on_page.append(np.asarray(resized_char, dtype=float))

    characters_on_page = np.asarray(get_hog(characters_on_page))

    predicted_styles = clf.predict(characters_on_page)

    archaic_counter = len(predicted_styles[predicted_styles==0])
    hasmonean_counter = len(predicted_styles[predicted_styles==1])
    herodian_counter = len(predicted_styles[predicted_styles==2])

    print("Individual Characters Style prediction counter")
    print('Archaic: ', archaic_counter, 'Hasmonean: ', hasmonean_counter, 'Herodian: ', herodian_counter)

    pred_style = max(archaic_counter, hasmonean_counter, herodian_counter)

    if pred_style == archaic_counter: return 'Archaic'
    if pred_style == hasmonean_counter: return 'Hasmonean'
    if pred_style == herodian_counter: return 'Herodian'


