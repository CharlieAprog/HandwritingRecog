import matplotlib.pyplot as plt
import numpy as np
import imageio
from skimage.feature import hog
from numpy import asarray
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import svm
import os


def get_hog(set):
    ##get histogram of oriented gradients for a data split set
    set = [hog(image, orientations=8, pixels_per_cell=(4, 4),
               cells_per_block=(1, 1)) for image in set]
    return set


def get_acc_SVM(dataset_train, dataset_test):
    name2idxStyle = {'archaic': 0, 'hasmonean': 1, 'herodian': 2}
    train_x, train_y = [], []
    # calculate all codebook vectors and store them in dict
    for stylename, styledataset in dataset_train.items():
        for label, characterset in styledataset.items():
            for char in characterset:
                train_x.append(np.asarray(char))
                train_y.append(name2idxStyle[stylename])

    train_x = np.asarray(get_hog(train_x))

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(train_x, train_y)

    test_x, test_y = [], []
    for stylename, styledataset in dataset_test.items():
        for label, characterset in styledataset.items():
            for char in characterset:
                test_x.append(np.asarray(char))
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


