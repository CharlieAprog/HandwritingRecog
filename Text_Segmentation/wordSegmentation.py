from skimage.filters import threshold_otsu
import numpy as np
from Text_Segmentation.lineSegmentation import timer


def extract_line_from_image(image, upper_line, lower_line):
    upper_boundary = np.min(upper_line[:, 0])
    lower_boundary = np.max(lower_line[:, 0])
    img_copy = np.copy(image)
    r, c = img_copy.shape
    x_axis = 0
    y_axis = 1
    for step in upper_line:
        img_copy[0:step[x_axis], step[y_axis]] = 0
    for step in lower_line:
        img_copy[step[x_axis]:r, step[y_axis]] = 0
    return img_copy[upper_boundary:lower_boundary, :]


def trim_line(line):
    thresh = threshold_otsu(line)
    line = line > thresh
    vertical_projection = np.sum(line, axis=0)
    line_threshold = 15
    b1 = 0
    b2 = 0
    beginning = 0
    end = 0
    temp1 = 0
    temp2 = 0

    for idx in range(len(vertical_projection)):
        if beginning == 0:
            if vertical_projection[idx] == 0:  # white
                if b1 <= line_threshold:
                    temp1 = 0
                    b1 = 0
            elif vertical_projection[idx] != 0:  # black
                if b1 == 0:  # start of black
                    temp1 = idx - 5 if idx - 5 > 0 else idx
                if b1 > line_threshold:
                    beginning = temp1
                b1 += 1

        if end == 0:
            idx2 = len(vertical_projection) - (idx + 1)
            if vertical_projection[idx2] == 0:  # white
                if b2 <= line_threshold:
                    temp2 = 0
                    b2 = 0
            elif vertical_projection[idx2] != 0:  # black

                if b2 == 0:  # start of black
                    temp2 = idx2 + 5 if idx + \
                        5 < len(vertical_projection) else idx2
                if b2 > line_threshold:
                    end = temp2
                b2 += 1

        if end != 0 and beginning != 0:
            break

    new_line = line[:, beginning:end]
    return new_line

@timer
def segment_words(line, vertical_projection):
    whitespace_lengths = []
    whitespace = 0

    # get whitespae lengths
    for idx in range(5, len(vertical_projection)-4):
        if vertical_projection[idx] == 0:
            whitespace = whitespace + 1
        elif vertical_projection[idx] != 0:
            if whitespace != 0:
                whitespace_lengths.append(whitespace)
            whitespace = 0  # reset whitepsace counter.
        if idx == len(vertical_projection)-1:
            whitespace_lengths.append(whitespace)

    print("whitespaces:", whitespace_lengths)
    avg_white_space_length = np.mean(whitespace_lengths)
    print("average whitespace lenght:", avg_white_space_length)

    # find index of whitespaces which are actually long spaces using the avg_white_space_length
    whitespace_length = 0
    divider_indexes = []
    divider_indexes.append(0)
    for index, vp in enumerate(vertical_projection[4:len(vertical_projection) - 5]):
        if vp == 0:  # white
            whitespace_length += 1
        elif vp != 0:  # black
            if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                divider_indexes.append(index-int(whitespace_length/2))
            whitespace_length = 0  # reset it
    divider_indexes.append(len(vertical_projection) - 1)
    divider_indexes = np.array(divider_indexes)
    dividers = np.column_stack((divider_indexes[:-1], divider_indexes[1:]))

    word_sizes = [np.sum(np.sum(line[:, window[0]:window[1]], axis=1))
                  for window in dividers]
    print(word_sizes)

    new_dividers = [window for window in dividers if np.sum(
        np.sum(line[:, window[0]:window[1]], axis=0)) > 200]

    return new_dividers