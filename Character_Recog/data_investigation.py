import numpy as np
import glob
import random
from PIL import Image
import cv2
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


def resize_pad(img,dim = 40):
## return image that is resized to a given shape without distorting it 
    height, width = img.shape
    # set new height and width similar to avg 
    height_all = dim
    width_all = dim
    #In case bounding boxes approach does not work
    
    if height == 0 or width == 0 :
        return None
    if height > width:
        resize_factor = height_all / height
        new_width = int(img.shape[1] * resize_factor)
        dim = (new_width, height_all)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        left_pad = int((width_all - new_width) / 2)
        right_pad = left_pad
        if left_pad + right_pad + new_width != dim:
            right_pad += 1
        resized_pad_img = cv2.copyMakeBorder(resized_img, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT,value = 255 )

    else:
        resize_factor = height_all / height
        new_height = int(img.shape[0] * resize_factor)
        dim = (width_all, new_height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        top_pad = int((height_all - new_height) / 2)
        bottom_pad = top_pad
        if top_pad + bottom_pad + new_height != dim:
            bottom_pad += 1
        resized_pad_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT,value = 255)

    return resized_pad_img


# input: path to some directory with images
# output: These images saved in a seperate directory according to their labels
def CropAndPadding(data_path):
    #This function crops and segments the images, removes the outliers
    data = glob.glob(data_path)
    data= random.sample(data,len(data))# Random shuffling for preprocessing inspection
    for img_name in data:

        #get label of image and crop images
        img_label = getlabel(img_name)
        img,im_bw = boundingboxcrop(img_name)

        # resize and pad without distorting character
        resized_pad_img = resize_pad(img,224)
        if resized_pad_img is None: continue

        #remove remaining noise from image if needed and save image
        img = noise_removal(resized_pad_img)
        saveimages(img_label,resized_pad_img)


# gives label only works on windows atm
def getlabel(img_name):
    #Regex for label extraction
    result = re.search(r'(?<=\\).+(?=\s*\\)', str(img_name))

    return result.group(0)


def saveimages(img_label,img):
    #saves the images into folders with labels
    directory = os.path.join('../data/grayscale_hhd_224x224', img_label)
    Path(directory).mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img)
    png = '.png'
    photoname = (img_label+'_') + str(uuid.uuid4())
    photoname = photoname + png

    img.save(os.path.join(directory, photoname), 'PNG')


def boundingboxcrop(img_name):
    ##crops characters to only their character coords in image
    im_bw = cv2.imread(img_name,0)
    retval, thresh_gray = cv2.threshold(im_bw, thresh=130, maxval=255,
                                        type=cv2.THRESH_BINARY)

    contours, image = cv2.findContours(thresh_gray, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0, 0, 0, 0)  # biggest bounding box so far
    mx_area = 0
    contours = contours[1:]
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h

        #get max area
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    x, y, w, h = mx

    return thresh_gray[y:y + h, x:x + w], im_bw[y:y + h, x:x + w]


def dataset_split(data_path):
# Accepts a data path and applies training /testing split (0.8 - 0.2)##
    splitfolders.ratio(data_path, output="style_train_test_has", seed=1337, ratio=(0.8, 0.2))


def noise_removal_grayscale(gray):
    #Removes contours of images that are not the character by masking
    graygray=gray.copy()
    resized_pad_img = gray.copy()
    retval, resized_pad_img = cv2.threshold(resized_pad_img.copy(), thresh=130, maxval=255,
                                   type=cv2.THRESH_BINARY)
    # Filter using contour area and remove small noise
    cnts, resized_pad_img = cv2.findContours(resized_pad_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)

    # ROI will be object with biggest contour
    cnt = cnt[2:]
    mask = np.ones(gray.shape[:2], dtype="uint8") * 255

    for c in cnt:
        cv2.drawContours(mask, [c], -1, 0, -1)
        cv2.drawContours(gray,[c]*255,-1,0,-1)
    newimg = cv2.bitwise_not((gray - mask) * 255)

    return newimg


def noise_removal(img,morphology=False):
    #Removes contours of images that are not the character by masking; 
    # also applies morphological erosion and dilation if needed
    resized_pad_img = img.copy()

    # Filter using contour area and remove small noise
    cnts, resized_pad_img = cv2.findContours(resized_pad_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)

    # ROI will be object with biggest contour
    cnt = cnt[2:]
    mask = np.ones(img.shape[:2], dtype="uint8") * 255

    for c in cnt:
        cv2.drawContours(mask, [c], -1, 0, -1)
    newimg = (img - mask) * 255
    retval, newimg = cv2.threshold(newimg, thresh=130, maxval=255,
                                   type=cv2.THRESH_BINARY_INV)

    if morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        newimg = cv2.morphologyEx(newimg, cv2.MORPH_CLOSE, kernel)

    return newimg