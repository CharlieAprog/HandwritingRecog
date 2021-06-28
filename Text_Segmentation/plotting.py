import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import os


def handle_saving(plotting_function):
    def wrapper_function(*args, **kwargs):
        plotting_function(*args, **kwargs)

        # save_args: save, folder_path, overwrite_path
        save_args = [x for x in kwargs.values()]

        if not save_args[0]:
            plt.show()
        else:
            if len(save_args) == 3:
                img_path = f"{save_args[1]}{os.path.sep}{plotting_function.__name__}.png"
                if not os.path.exists(
                        f"{save_args[1]}{os.path.sep}{plotting_function.__name__}.png"
                ):
                    plt.savefig(img_path)
                    print(
                        f"Image {os.path.basename(img_path)} has been saved. Overwriting=0."
                    )
                elif save_args[2]:
                    plt.savefig(img_path)
                    print(
                        f"Image {os.path.basename(img_path)} has been saved. Overwriting=1."
                    )
                else:
                    print(f"Image path already exists: [{img_path}]")
            else:
                # No other cases are needed to be taken into account, as 'save' and 'folder_path' are trivial to be
                # provided when one wishes to save an image, overwrite is the only thing that may easily be forgotten
                raise IndexError(
                    f"Expected number of arguments: 3 (save, folder_path, overwrite) but received only 2"
                )
    return wrapper_function


@handle_saving
def plot_hist(hist,
              y_threshold,
              save=False,
              folder_path=None,
              overwrite_path=False):
    fs = 25
    plt.figure(figsize=(10, 6))
    plt.plot(hist)
    plt.axhline(y=y_threshold, color="r", linestyle="-")
    plt.ylim(0, max(hist) * 1.1)
    plt.xlabel("Row", fontsize=fs)
    plt.ylabel("Black pixels", fontsize=fs)
    plt.title("Binary image black pixel counting result", fontsize=fs)
    plt.yticks(fontsize=fs - 5)
    plt.xticks(fontsize=fs - 5)
    plt.grid()


@handle_saving
def plot_hist_lines_on_image(newImage,
                             midlines,
                             save=False,
                             folder_path=None,
                             overwrite_path=False):
    plt.figure(figsize=(10, 6))
    plt.imshow(newImage, cmap="gray")
    for i in range(len(midlines)):
        for idx, loc in enumerate(midlines[i]):
            if idx == 0:
                plt.axhline(y=loc, color="r", linestyle="-")
            else:
                plt.axhline(y=loc, color="b", linestyle="-")


@handle_saving
def plot_paths_next_to_image(binary_image,
                             paths,
                             save=False,
                             folder_path=None,
                             overwrite_path=False):
    fig, ax = plt.subplots(figsize=(16, 12), ncols=2)
    for path in paths:
        path = np.array(path)
        ax[1].plot((path[:, 1]), path[:, 0])
    ax[1].axis("off")
    ax[0].axis("off")
    ax[1].imshow(binary_image, cmap="gray")
    ax[0].imshow(binary_image, cmap="gray")


@handle_saving
def plot_hough_transform(hspace,
                         theta,
                         dist,
                         x0,
                         x1,
                         origin,
                         image,
                         newImage,
                         save=False,
                         folder_path=None,
                         overwrite_path=False):
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))
    ax = axes.ravel()

    # Axis 0
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    # # Axis 1
    # ax[1].imshow(np.log(1 + hspace),
    #              extent=[
    #                  np.rad2deg(theta[-1]),
    #                  np.rad2deg(theta[0]), dist[-1], dist[0]
    #              ],
    #              cmap='gray',
    #              aspect=1 / 1.5)
    # ax[1].set_title('Hough transform')
    # ax[1].set_xlabel('Angles (degrees)')
    # ax[1].set_ylabel('Distance (pixels)')
    # ax[1].axis('image')

    # Axis 2
    ax[2].imshow(image, cmap='gray')
    ax[2].plot(origin, (x0, x1), '-b')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    # Axis 3
    ax[3].imshow(newImage, cmap='gray')

    plt.tight_layout()


def plot_simple_images(image_list, title=None):
    if type(image_list) != list:
        image_list = [image_list]
    fig, ax = plt.subplots(nrows=len(image_list), figsize=(5, 6))
    for index, image in enumerate(image_list):
        if len(image_list) > 1:
            ax[index].imshow(image, cmap="gray")
        else:
            ax.imshow(image, cmap="gray")
    fig.suptitle(title) if title else fig.suptitle('')
    plt.show()


def plot_words_in_line(line, words):
    fig, ax = plt.subplots(nrows=len(words), figsize=(5, 8))
    for index, word in enumerate(words):
        ax[index].axis("off")
        ax[index].imshow(word, cmap="gray")
    plt.show()


def plot_grid(images, title=None):
    # distribute bottom row images
    bottom_images = images[2:]
    axes = []
    fig = plt.figure()
    # ---------- odd amount of images ----------
    if len(bottom_images) % 2:
        bottom_first_col = bottom_images[:len(bottom_images) // 2]
        bottom_second_col = bottom_images[len(bottom_images) // 2:-1]
        total_col_num = 2 + len(bottom_first_col) + 1
    # ---------- even amount of images ----------
    else:
        bottom_first_col = bottom_images[:len(bottom_images) // 2]
        bottom_second_col = bottom_images[len(bottom_images) // 2:]
        total_col_num = 2 + len(bottom_first_col)
    # grid: first two rows
    for i in range(2):
        axes.append(plt.subplot2grid((total_col_num, 2), (i, 0), colspan=2, rowspan=round(total_col_num * 0.25)))
    # grid: last row
    for idx, image in enumerate(bottom_first_col):
        axes.append(plt.subplot2grid((total_col_num, 2), (2 + idx, 0), colspan=1))
    for idx, image in enumerate(bottom_second_col):
        axes.append(plt.subplot2grid((total_col_num, 2), (2 + idx, 1), colspan=1))
    # plot all rows
    print(len(axes), len(images))
    for idx, ax in enumerate(axes):
        if len(images[idx].shape) == 2:
            ax.imshow(images[idx])
        else:
            ax.plot(images[idx])
    plt.title(title) if title else plt.title(' ')
    plt.show()


def plot_connected_component_label(path):
    """
    Plots the connected components of an image file.
    In our system it should take a binary image (black background, white foreground) as 'path'.
    """
    # Getting the input image
    img = cv2.imread(path, 0)
    # Converting those pixels with values 1-127 to 0 and others to 1
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    print(type(img))
    print(type(img[0]), img[0])
    # Applying cv2.connectedComponents()
    num_labels, labels = cv2.connectedComponents(img)

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    # Showing Original Image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Orginal Image")
    plt.show()

    # Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()


def plot_connected_component_bounding_boxes(image, rectangle_boundaries, title=None):
    plt.imshow(image, cmap="gray")
    ax = plt.gca()
    for idx, box in enumerate(rectangle_boundaries):
        if idx == 0:
            print(box)
        y_min = box[0][1]
        y_max = box[0][0]
        x_min = box[1][0]
        x_max = box[1][1]
        width = x_max - x_min
        height = y_max - y_min
        ax.add_patch(
            Rectangle((x_min, y_min),
                      width,
                      height,
                      linewidth=1,
                      edgecolor='r',
                      facecolor='none'))
    plt.show()
