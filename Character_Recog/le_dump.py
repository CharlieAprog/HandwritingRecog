import matplotlib.pyplot as plt


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
    print('img:', img)
    print('img:size', img.shape)
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
    fig, ax = plt.subplots(2, 2)  # sharex='col', sharey='row')
    ax[0, 1].hist(hist_row, orientation='horizontal', align='mid', range={0, len(hist_row)}, bins=len(hist_row))
    ax[1, 0].hist(hist_col, range={0, len(hist_col)}, bins=len(hist_col))
    ax[0, 0].imshow(img_np, cmap="gray")
    plt.show()

'''
        #Keep data path of outlier:
        if (height >= avg_y +4*std_y) or (width >= avg_x + 4*std_x):
            outliers_number+=1
            outliers_paths.append(img_name)
            #axarr[0,1].imshow(roi,cmap='gray')
            #plt.show()
            #print(img_name)
        # if (height/width) >= 1.5  or (width/height) >= 1.5:
        #     f, axarr = plt.subplots(2, 2)
        #     axarr[0, 1].imshow(roi, cmap='gray')
        #     plt.show()
        #     print(img_name)
        #     print('height/width outlier')
    print('Number of outliers to be removed:',outliers_number)
    #extract max dimensions after outlier removal for padding
    max_x = 0
    max_y = 0
    for img_name in data:
        if img_name not in outliers_paths:
            roi = boundingboxcrop(img_name)
            roi = thresh_gray[y:y + h, x:x + w]
            height, width = roi.shape
            #extract max dimensions for padding
            if height > max_y:
                max_y = height
            if width > max_x:
                max_x = width
    print(max_x,max_y,'new max dimensions after outlier removal.')

    #now pad according to max dimensions and save
    for img_name in data:
        if img_name not in outliers_paths:
            img_label =  getlabel(img_name)
            roi = boundingboxcrop(img_name)
            height, width = roi.shape
            #pad image according to max dimensions after outlier removal
            paddedroi = cv2.copyMakeBorder(roi, max_y - height, 0, 0,max_x-width, cv2.BORDER_CONSTANT, 255)
            #print comparison between padded and original segmented image
            #f, axarr = plt.subplots(2, 2)
            #axarr[0,0].imshow(paddedroi,cmap='gray')
            #axarr[0,1].imshow(roi,cmap='gray')
            #plt.show()

            #save processed images in new folder
            saveimages(img_label,paddedroi)
        #print new max dimension
        #print(max_y,max_x,' These are the max dimensions')

'''
