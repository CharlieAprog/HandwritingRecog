import numpy as np
from Text_Segmentation.lineSegmentation import timer


def trim_section(section, section_threshold=10):
    """ Function for removing padding from left and right side of a character section """
    vertical_projection = np.sum(section, axis=0)
    b1 = 0
    b2 = 0
    beginning = 0
    end = 0
    temp1 = 0
    temp2 = 0

    for idx in range(len(vertical_projection)):
        if beginning == 0:
            if vertical_projection[idx] == 0:  # white
                if b1 <= section_threshold:
                    temp1 = 0
                    b1 = 0
            elif vertical_projection[idx] != 0:  # black
                if b1 == 0:  # start of black
                    temp1 = idx - 1 if idx - 1 > 0 else idx
                if b1 > section_threshold:
                    beginning = temp1
                b1 += 1

        if end == 0:
            idx2 = len(vertical_projection) - (idx + 1)
            if vertical_projection[idx2] == 0:  # white
                if b2 <= section_threshold:
                    temp2 = 0
                    b2 = 0
            elif vertical_projection[idx2] != 0:  # black

                if b2 == 0:  # start of black
                    temp2 = idx2 + 1 if idx + \
                                        1 < len(vertical_projection) else idx2
                if b2 > 10:
                    end = temp2
                b2 += 1

        if end != 0 and beginning != 0:
            break

    new_section = section[:, beginning:end]
    return new_section.astype(np.uint8)


def segment_words(section, vertical_projection):
    whitespace_lengths = []
    whitespace = 0

    # A) Get whitespace lengths in the more dense subsection of a section, hence ranging from 5 to len(v_p)-4, to avoid
    # the bias of long ascenders and descenders
    for idx in range(5, len(vertical_projection) - 4):
        if vertical_projection[idx] == 0:
            whitespace = whitespace + 1
        elif vertical_projection[idx] != 0:
            if whitespace != 0:
                whitespace_lengths.append(whitespace)
            whitespace = 0  # reset whitespace counter.
        if idx == len(vertical_projection) - 1:
            whitespace_lengths.append(whitespace)
    # print("whitespaces:", whitespace_lengths)
    avg_white_space_length = np.mean(whitespace_lengths)
    # print("average whitespace lenght:", avg_white_space_length)

    # B) Find words with whitespaces which are actually long spaces (word breaks) using the avg_white_space_length
    whitespace_length = 0
    divider_indexes = [0]
    for index, vp in enumerate(vertical_projection[4:len(vertical_projection) - 5]):
        if vp == 0:  # white
            whitespace_length += 1
        elif vp != 0:  # black
            if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                divider_indexes.append(index - int(whitespace_length / 2))
            whitespace_length = 0  # reset it
    divider_indexes.append(len(vertical_projection) - 1)
    divider_indexes = np.array(divider_indexes)
    dividers = np.column_stack((divider_indexes[:-1], divider_indexes[1:]))
    new_dividers = [window for window in dividers if np.sum(
        np.sum(section[:, window[0]:window[1]], axis=0)) > 200]

    return new_dividers


def trim_360(image, section_thresh=5):
    """ Returns an image with no padding on either side """
    trim1 = trim_section(np.rot90(image).astype(int), section_threshold=section_thresh)
    trim2 = trim_section(np.rot90(trim1, axes=(1, 0)).astype(int), section_threshold=section_thresh - 5)
    return trim2


def word_segmentation(section_images):
    words_in_sections = []  # |-------- pad obtained sections
    sections = []
    for idx in range(len(section_images)):
        section = trim_section(section_images[idx])
        if section.shape[0] == 0 or section.shape[1] == 0:
            continue
        sections.append(section)

        vertical_projection = np.sum(section, axis=0)
        dividers = segment_words(section, vertical_projection)
        words = []
        for window in dividers:
            word = section[:, window[0]:window[1]]
            trimmed_word = trim_360(word)
            # plotSimpleImages(sliding_words[-1])
            words.append(trimmed_word)
        words_in_sections.append(words)
        images = [section, vertical_projection]
        images.extend(words)
        # plotGrid(images_boolean_to_binary)
        # plotGrid(images)
    print("Word segmentation complete.")
    return sections, words_in_sections