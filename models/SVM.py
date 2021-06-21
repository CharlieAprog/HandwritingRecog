import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
def get_numpy_arrays(dict,map):
    x,y = [],[]
    for label, item in dict.items():
        for image in item: 
            y.append(map[label])
            x.append(image)

    return x,y

def get_hog(set):
    set = [hog(image,orientations=8, pixels_per_cell=(6, 6),
                    cells_per_block=(1, 1)) for image in set]
    return set


def get_char_images(style_path: str, character: str):
    """ Returns a list of numpy arrays, where each arrays corresponds to an image from a certain character of a
    certain class """
    list_for_glob = style_path + character + '/*.png'
    img_name_list = glob.glob(list_for_glob)
    img_list = [cv2.imread(img, 0) for img in img_name_list]
    assert len(img_list) > 0, "Trying to read image files while being in a wrong folder."
    return img_list

characters = ['Alef','Ayin','Bet','Dalet','Gimel','He','Het','Kaf','Kaf-final','Lamed','Mem','Mem-medial',
            'Nun-final','Nun-medial','Pe','Pe-final','Qof','Resh','Samekh','Shin'
            ,'Taw','Tet','Tsadi-final','Tsadi-medial','Waw','Yod','Zayin']

character_map = {'Alef':0,'Ayin':1,'Bet':2,'Dalet':3,'Gimel':4,'He':5,'Het':6,'Kaf':7,'Kaf-final':8,'Lamed':9,
            'Mem':10,'Mem-medial':11,'Nun-final':12,'Nun-medial':13,'Pe':14,'Pe-final':15,'Qof':16,'Resh':17,
            'Samekh':18,'Shin':19,'Taw':20,'Tet':21,'Tsadi-final':22,'Tsadi-medial':23,'Waw':24,'Yod':25,'Zayin':26}

path =  'C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/data/Char_Recog/binarized_monkbrill_split_40x40_morphed/'
train_path = path + 'train/'
test_path = path + 'val/'
test_path_nomorph = path +'val_no_morph/'

train_set = {char:
                [img for img in get_char_images(train_path, char)]
                for char in characters}
test_set = {char:
                [img for img in get_char_images(test_path, char)]
                for char in characters}
test_set_nomorph = {char:
                [img for img in get_char_images(test_path_nomorph, char)]
                for char in characters}

trainx,trainy=  get_numpy_arrays(train_set,character_map)
testx,testy=  get_numpy_arrays(test_set,character_map)
testx_nomorph,testy_nomorph = get_numpy_arrays(test_set_nomorph,character_map)  

trainy = np.asarray(trainy).flatten()
testy = np.asarray(testy).flatten()
testy_nomorph = np.asarray(testy_nomorph).flatten()
trainx = np.asarray(get_hog(trainx))
testx= np.asarray(get_hog(testx))
testx_nomorph = np.asarray(get_hog(testx_nomorph))

pca = PCA(n_components=200)
pca.fit(trainx)

trainx= pca.transform(trainx)
testx =pca.transform(testx)
testx_nomorph = pca.transform(testx_nomorph)

clf = svm.SVC()
clf.fit(trainx, trainy)
predict =  clf.predict(testx_nomorph)

print('accuracy on test no morph set:',accuracy_score(testy_nomorph,predict))
plot_confusion_matrix(clf,X=testx_nomorph, y_true= testy_nomorph)
plt.show()