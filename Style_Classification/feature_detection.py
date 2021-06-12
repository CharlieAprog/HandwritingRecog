import cv2
from dim_reduction import get_style_char_images
import matplotlib.pyplot as plt
import numpy as np
import sys

def get_hinge(img_label,archaic_imgs,hasmonean_imgs,herodian_imgs):
    #given a time period, calculate the Hinge histogram occurences of phi1 and phi2
    '''
    sample img to develop function
    sample_img = herodian_imgs[img_label][0]
    Gaussian blurring w/ 5x5 kernel for better edge detection
    blurred_image  = cv2.GaussianBlur(img,(5,5),0)
    Edges = cv2.Canny(blurred_image,0,100)
    show edges found
    imgplot = plt.imshow(Edges)
    plt.show()
    '''
    for images in herodian_imgs[img_label]:
        images =cv2.bitwise_not(images)
        #print(images)
        blurred_image  = cv2.GaussianBlur(images,(5,5),0)
        Edges = cv2.Canny(blurred_image,0,100)
        thresh = cv2.threshold(images, 30, 255, cv2.THRESH_BINARY)[1]
        thresh= cv2.Canny(thresh,0,100)


        corners = cv2.goodFeaturesToTrack(thresh,4,0.01,50,useHarrisDetector=True,k=0.04)
        imagesrgb=cv2.cvtColor(images,cv2.COLOR_GRAY2RGB)
        print("Corners:")
        for c in corners:
            x,y = c.ravel()
            cv2.circle(imagesrgb,(x,y),3,(255,0,),-1)
        
        cv2.imshow("binary image",thresh)
        cv2.imshow("iamge/cornersdrawn", imagesrgb)
        cv2.waitKey(0)





    #---TO-DO:---#
    #1)create def get_textural_feature
    #2) get image, get image label from CNN
    #3) create SOM:
    #   3)a)get all specific characters from three periods given the CNN label
    #   4)b) extract allographic and textural features from all these characters
    #   4)c) apply PCA on vectors from 4)b) (if needed)
    #   ..

style_base_path = './data/characters_for_style_classification_balance_morph/'
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
new_size = (400, 400)  # change this to something which is backed up by a reason
archaic_imgs = {char:
                    [cv2.resize(img, new_size).flatten() for img in get_style_char_images(style_archaic_path, char)]
                    for char in archaic_characters}
hasmonean_imgs = {char:
                    [cv2.resize(img, new_size).flatten() for img in get_style_char_images(style_hasmonean_path, char)]
                    for char in hasmonean_characters}
herodian_imgs = {char:
                    [cv2.resize(img, new_size).flatten() for img in get_style_char_images(style_herodian_path, char)]
                    for char in herodian_characters}

archaic_imgs_unflattened = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_archaic_path, char)]
                    for char in archaic_characters}
hasmonean_imgs_unflattened = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_hasmonean_path, char)]
                    for char in hasmonean_characters}
herodian_imgs_unflattened = {char:
                    [cv2.resize(img, new_size) for img in get_style_char_images(style_herodian_path, char)]
                    for char in herodian_characters}

    #show img
    #imgplot = plt.imshow(herodian_imgs['Alef'][0])
    #plt.show()

img_label = 'Alef'
    #get hinge histogram
hingehist_archaic,hingehist_hasmonean,hingehist_herodian = get_hinge(img_label,archaic_imgs_unflattened,hasmonean_imgs_unflattened,herodian_imgs_unflattened)
    
