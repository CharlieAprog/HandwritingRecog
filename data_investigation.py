import numpy as np 
import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize

# input Image file 
# output binarized image as np array
def binarize(img):
    img_np = np.asarray(img)
    # plt.imshow(img_np)
    # plt.show()
    # choose where you wanna set binary threshold
    threshold = 120
    binary_img = np.empty([len(img_np[:, 0]), len(img_np[0, :])])
    dummy_col = np.empty(len(img_np[:, 0]))
    for idx in range(0, len(img_np[0, :])):
        col = img_np[:, idx]
        for x in range(len(col)):
            if col[x] < threshold:
                dummy_col[x] = 0
            else:
                dummy_col[x] = 1
        binary_img[:, idx] = dummy_col
    # plt.imshow(binary_img)
    # plt.show()
    return binary_img

# input: image file
# output: creates plot with image and histograms next to it
# also this plot is made for the binarized image
def plot_hist(img):
    print('img:',img)
    print('img:size',img.shape)
    img_np = np.asarray(img)
    print(img_np.shape)
    hist_row = []
    hist_col = []
    print(len(img_np[0, :]))
    print(len(img_np[:, 0]))
    for idx in range(0, len(img_np[0, :])):
        col_arr = img_np[:, idx]
        # get the amount of black pixels
        arr = col_arr[col_arr == 0]
        hist_col.append(len(arr))
    for idx in range(len(img_np[:, 0])):
        row_arr = img_np[idx, :]
        # get the amount of black pixels
        arr = row_arr[row_arr == 0]
        hist_row.append(len(arr))
    print(len(hist_col))
    print(len(hist_row))
    fig, ax = plt.subplots(2, 2) #sharex='col', sharey='row')
    ax[0,1].hist(hist_row, orientation='horizontal', align='mid', range={0, len(hist_row)}, bins=len(hist_row))
    ax[1,0].hist(hist_col, range={0, len(hist_col)},bins=len(hist_col))
    ax[0,0].imshow(img_np, cmap="gray")
    plt.show()
 

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
        # plot histograms when one of dimmensions is larger then avg + std
        # comment when no histogram wanted
        #if img_size[0] > 43 or img_size[1] > 56:
           # plot_hist(binarize(img))
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
    # max_imgx.show()
    # max_imgy.show()
    return avg_x,avg_y

def CroppedCharAnalysis(data_path):
    data = glob.glob(data_path)
    #use to find std, mean of newly cropped images.
    img_size_all_x = 0
    img_size_all_y = 0
    std_y = 0
    std_x = 0
    max_x = 0
    max_y = 0
    for img_name in data:
        im_bw = cv2.imread(img_name,0) #Grayscale conversion
        height, width = im_bw.shape
        retval, thresh_gray = cv2.threshold(im_bw, thresh=100, maxval=255, \
                                            type=cv2.THRESH_BINARY_INV)

        contours, image= cv2.findContours(thresh_gray, cv2.RETR_LIST, \
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

        roi = thresh_gray[y:y + h, x:x + w]
        ##Extract average column and row sizes
        height,width = roi.shape
        img_size_all_y += height
        img_size_all_x += width
        std_y += abs(height - (img_size_all_y / len(data)))
        std_x += abs(width - (img_size_all_x / len(data)))
        if height > max_y:
            max_imgy = img
            max_y = height
        if width > max_x:
            max_imgx = img
            max_x = width
    std_y = std_y / len(data)
    std_x = std_x / len(data)
    avg_x = img_size_all_x / len(data)
    avg_y = (img_size_all_y / len(data))
    print('---After cropping analysis---')
    print("Max image dimmension: ", max_x, max_y)
    print("Average image dimension: ", avg_x, avg_y)
    print("Standard deviation of dimensions: ", std_x, std_y)
    return avg_x, avg_y, std_x, std_y,max_x,max_y

def FindLargestContour(data_path,avg_x,avg_y):
    data = glob.glob(data_path)
    image_no = 1

    #use to find std, mean of newly cropped images.
    img_size_all_x = 0
    img_size_all_y = 0
    std_y = 0
    std_x = 0
    for img_name in data:
        im_bw = cv2.imread(img_name,0) #Grayscale conversion
        height, width = im_bw.shape
        retval, thresh_gray = cv2.threshold(im_bw, thresh=100, maxval=255, \
                                            type=cv2.THRESH_BINARY_INV)

        contours, image= cv2.findContours(thresh_gray, cv2.RETR_LIST, \
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


        roi = thresh_gray[y:y + h, x:x + w]

      # roiresize = resize(roi,(int(avg_x),int(avg_y)))
        #roipadded =



        f, axarr = plt.subplots(2, 2)
        axarr[0,0].imshow(roiresize,cmap='gray')
        axarr[0,1].imshow(roi,cmap='gray')
        print(roi.shape,'aaa')
        plt.show()

        #First save for extracting data info
        #name = 'C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/data/Cropped/' + str(image_no) + '.PGM'
        #im = Image.fromarray(roi)
        #im.save(name, 'PGM')
        #image_no += 1





#Data investigation before crop
avg_x,avg_y = data_size_stats("data/monkbrill/*/*.pgm")
#Get stats for cropped images
avg_x, avg_y,std_x,std_y,max_x,max_y = CroppedCharAnalysis("data/monkbrill/*/*.pgm")
#crop ,resize,save images
FindLargestContour("data/monkbrill/*/*.pgm",avg_x,avg_y,max_x,max_y)




