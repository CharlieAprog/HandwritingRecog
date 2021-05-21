import numpy as np
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
#from test import create_image
import re
from pathlib import Path
import os
import uuid
import splitfolders

# input: path to some image data
# output: print size stats of the images
# also plots histograms for all outlier images atm -> comment if not wanted
def data_size_stats(data_path):
    data = glob.glob(data_path)
    # get mean + max dimmensions
    img_size_all_x = 0
    img_size_all_y = 0
    max_x = 0
    max_y = 0
    for img_name in data:
        img = Image.open(img_name)
        img_size = img.size
        img_size_all_y += img_size[0]
        img_size_all_x += img_size[1]
        if img_size[0] > max_y:
            max_imgy = img
            max_y = img_size[0]
        if img_size[1] > max_x:
            max_imgx = img
            max_x = img_size[1]
    # get std of dimmensions
    std_y = 0
    std_x = 0
    for img_name in data:
        img = Image.open(img_name)
        img_size = img.size
        std_y += abs(img_size[0]-(img_size_all_y/len(data)))
        std_x += abs(img_size[1]-(img_size_all_x/len(data)))
    std_y = std_y / len(data)
    std_x = std_x / len(data)
    avg_x = img_size_all_x/len(data)
    avg_y = (img_size_all_y/len(data))
    print("Max image dimmension: ", max_x, max_y)
    print("Average image dimmension: ", avg_x,  avg_y)
    print("Standart deviation of dimmensions: ", std_x, std_y)
    # only works for one at the time -> print the max outliers
    return avg_x,avg_y

# This function crops images and then finds the stats of the image dimmensions
def CroppedCharAnalysis(data_path):
    #gets the mean of the cropped data, will be used for finding the std
    data = glob.glob(data_path)
    img_size_all_x = 0
    img_size_all_y = 0
    max_x = 0
    max_y = 0
    for img_name in data:
        roi,im_bw = boundingboxcrop(img_name)
        height,width = roi.shape
        img_size_all_y += height
        img_size_all_x += width
        #get max dimnesiosn before contour analysis
        if height > max_y:
            max_y = height
        if width > max_x:
            max_x = width
    # get avg dimesiom before contour analysis
    avg_x = img_size_all_x / len(data)
    avg_y = (img_size_all_y / len(data))
    print('---After cropping analysis---')
    print("Max image dimmension: ", max_x, max_y)
    print("Average image dimension: ", avg_x, avg_y)
    return max_x,max_y,img_size_all_x,img_size_all_y

# Main function
# input: path to some directory with images
# output: These images saved in a seperate directory according to their labels
def CropAndPadding(data_path):
    #This function crops and segments the images, removes the outliers
    data = glob.glob(data_path)
    data= random.sample(data,len(data))# Random shuffling for preprocessing inspection
    for img_name in data:
        img_label = getlabel(img_name)
        img,im_bw = boundingboxcrop(img_name)
        # here we have the cropped images
        # roi is image name height, width of that image
        height, width = img.shape
        # set new height and width similar to avg but now I make sure we have square images
        height_all = 224
        width_all = 224
        #In case bounding boxes approach does not work
        if height == 0 or width == 0 :
            continue
        if height > width:
            resize_factor = height_all / height
            new_width = int(img.shape[1] * resize_factor)
            dim = (new_width, height_all)
            resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
            left_pad = int((width_all - new_width) / 2)
            right_pad = left_pad
            if left_pad + right_pad + new_width != 224:
                right_pad += 1
            resized_pad_img = cv2.copyMakeBorder(resized_img, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT,value = 255 )
        else:
            resize_factor = width_all / width
            new_height = int(img.shape[0] * resize_factor)
            dim = (width_all, new_height)
            resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
            top_pad = int((height_all - new_height) / 2)
            bottom_pad = top_pad
            if top_pad + bottom_pad + new_height != 224:
                bottom_pad += 1
            resized_pad_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT,value = 255)
       # Plotting code; Keep for debugging
       # f, axarr = plt.subplots(2, 2)
        #axarr[0, 0].imshow(im_bw)
        #axarr[0, 1].imshow(resized_pad_img)
        #plt.show()
        #print(resized_pad_img.shape)
        saveimages(img_label,resized_pad_img)

def findstd(data_path,avg_x,avg_y,img_size_all_x,img_size_all_y):
    #get the standard devation  of the data after the cropping
    data = glob.glob(data_path)
    std_y = 0
    std_x = 0
    for img_name in data:
        roi,im_bw = boundingboxcrop(img_name)
        height, width = roi.shape
        std_y += abs(height - (img_size_all_y / len(data)))
        std_x += abs(width - (img_size_all_x / len(data)))

    std_y = std_y / len(data)
    std_x = std_x / len(data)

    print('Standard deviation of cropped pictures for x and y:')
    print( std_x, std_y )
    return std_x,std_y

# gives label only works on windows atm
def getlabel(img_name):
    #Regex for label extraction
    #print(img_name)
    result = re.search(r'(?<=\\).+(?=\s*\\)', str(img_name))
    #print('extracted:',result.group(0))
    return result.group(0)

def saveimages(img_label,img):
    #saves the images into folders
    directory = os.path.join('../data/grayscale_monkbrill_split_224x224', img_label)
    Path(directory).mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img)
    png = '.png'
    photoname = (img_label+'_') + str(uuid.uuid4())
    photoname = photoname + png
    img.save(os.path.join(directory, photoname), 'PNG')

def boundingboxcrop(img_name):
    im_bw = cv2.imread(img_name,0)  # Grayscale conversion
    retval, thresh_gray = cv2.threshold(im_bw, thresh=130, maxval=255,
                                        type=cv2.THRESH_BINARY_INV)

    contours, image = cv2.findContours(thresh_gray, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0, 0, 0, 0)  # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    x, y, w, h = mx
    return im_bw[y:y + h, x:x + w],im_bw

#input: data_path where preprocessed data are
#output: data are split into 70-30 training testing split
def dataset_split(data_path):
    splitfolders.ratio(data_path, output="binarized_monkbrill_split_40x40", seed=1337, ratio=(.7, 0.3))
