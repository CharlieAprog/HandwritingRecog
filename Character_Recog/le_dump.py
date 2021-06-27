import matplotlib.pyplot as plt

# input Image file
# output binarized image as np array
def binarize(img):
    img_np = np.asarray(img)

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
    #plot histogram
    fig, ax = plt.subplots(2, 2)  # sharex='col', sharey='row')
    ax[0, 1].hist(hist_row, orientation='horizontal', align='mid', range={0, len(hist_row)}, bins=len(hist_row))
    ax[1, 0].hist(hist_col, range={0, len(hist_col)}, bins=len(hist_col))
    ax[0, 0].imshow(img_np, cmap="gray")

    plt.show()
