from Text_Segmentation.plotting import *
from Text_Segmentation.textSegmentation import calc_outlier
from Text_Segmentation.textSegmentation import trim_360
import itertools
import copy


def slide_over_word(word, window_size, shift):
    images = []
    height, width = word.shape
    for snap in range(0, width - window_size, shift):
        images.append(word[:, snap:snap + window_size])
        # plotSimpleImages([word[:, snap:snap + window_size]])
    images.append(word[:, word.shape[1] - window_size : word.shape[1]])
    return images


def get_sliding_words(words_in_lines, window_size, shift):
    sliding_words_in_line = []
    for line in words_in_lines:
        sliding_words = []
        for word in line:
            sliding_words.append(slide_over_word(word, window_size, shift))
        sliding_words_in_line.append(sliding_words)
    return sliding_words_in_line


def getComponentClusters(num_labels, labels):
    clusters = [[] for _ in range(num_labels)]
    for row_idx, row in enumerate(labels):
        for col_idx, col in enumerate(row):
            clusters[col].append([row_idx, col_idx])
    del clusters[0]
    return clusters


def getBoundingBoxBoundaries(image, clusters):
    box_boundaries = []
    for idx, cluster in enumerate(clusters):
        # initialize starting values
        y_max, y_min, x_max, x_min = [image.shape[0], 0, 0, image.shape[1]]

        for coordinate in cluster:
            if coordinate[0] < y_max:
                y_max = coordinate[0]
            elif coordinate[0] > y_min:
                y_min = coordinate[0]
            if coordinate[1] > x_max:
                x_max = coordinate[1]
            elif coordinate[1] < x_min:
                x_min = coordinate[1]
        if x_max != image.shape[1]:
            x_max += 1
        if x_min != 0:
            x_min -= 1
        box_boundaries.append([[y_max, y_min], [x_min, x_max]])
    return box_boundaries


def dialate_clusters(num_boxes, word):
    kernel = np.ones((5, 3), np.uint8)
    word = cv2.dilate(word, kernel, iterations=1)
    num_labels, labels = cv2.connectedComponents(word)
    clusters = getComponentClusters(num_labels, labels)
    box_boundaries = getBoundingBoxBoundaries(word, clusters)
    # plotConnectedComponentBoundingBoxes(word, box_boundaries)
    return box_boundaries, word


def get_box_images(box_boundaries, word):
    """Returns all the bounded images and areas thereof within a word that can contain any number of characters"""
    box_images = []
    box_areas = []
    for box in box_boundaries:
        y_min = box[0][0]
        y_max = box[0][1]
        x_min = box[1][0] - 1
        x_max = box[1][1] + 1
        box_img = word[y_min:y_max, x_min:x_max]
        box_areas.append(abs(y_max - y_min) * abs(x_max - x_min))
        box_images.append(box_img)
    return box_images, box_areas


def character_segment(word, title=None):
    cluster_threshold = 10
    word = word.astype(np.uint8)
    print("Running character segmentation...")
    num_labels, labels = cv2.connectedComponents(word)
    clusters = getComponentClusters(num_labels, labels)
    box_boundaries = getBoundingBoxBoundaries(word, clusters)
    num_boxes = len(box_boundaries)
    while num_boxes > cluster_threshold:
        box_boundaries, word = dialate_clusters(num_boxes, word)
        num_boxes = len(box_boundaries)
        print(num_boxes)
    box_images, box_areas = get_box_images(box_boundaries, word)
    # plotConnectedComponentBoundingBoxes(word, box_boundaries, title = title)
    print("Character segmentation complete.")
    return box_images, box_areas, word, box_boundaries


def run_character_segment(words_in_lines):
    segmented_word_box_images = []
    segmented_word_box_areas = []
    all_box_boundaries = []
    for line_idx, line in enumerate(words_in_lines):
        line_word_images = []
        line_word_areas = []
        box_boundaries_lines = []
        for word_idx, word in enumerate(line):
            pixels, areas, new_word, box_boundaries = character_segment(word, title="[OLD]")
            words_in_lines[line_idx][word_idx] = new_word
            line_word_images.append(pixels)
            line_word_areas.append(areas)
            box_boundaries_lines.append(box_boundaries)
        segmented_word_box_images.append(line_word_images)
        segmented_word_box_areas.append(line_word_areas)
        all_box_boundaries.append(box_boundaries_lines)
    return segmented_word_box_images, segmented_word_box_areas, all_box_boundaries

# def single_char_clean(character):
#     pixels, areas, new_word, box_boundaries = character_segment(character, title="[OLD]")
#     min_area = 500
#     max_area = 8000  # anything above 8000 is undoubtedly more than 1 character in any test image
#     area_thr = lambda img, x: [x for x in img if x > min_area and x < max_area]
#     boxes_thr = [area_thr(image, None) for image in areas]

   
    



def is_boundary_included(all_boundries, cluster):
    x_min_input = cluster[1][0]
    x_max_input = cluster[1][1]
    for idx, boundries in enumerate(all_boundries):
        x_min_current = boundries[1][0]
        x_max_current = boundries[1][1]
        if x_min_input > x_min_current and x_max_input < x_max_current:
            return True
    return False


def is_image_border_active(character):
    left_border = character[:, 0]
    right_border = character[:, -1]
    if len(np.nonzero(left_border)[0]) > 0 or len(np.nonzero(right_border)[0]) > 0:
        return True
    return False


def filter_characters(segmented_word_box_areas, segmented_word_box_images, all_box_boundaries, words_in_lines):
    """
    Returns those images that are supposedly not artifacts.
    """
    # Empirically observed values
    min_area = 500
    max_area = 8000  # anything above 8000 is undoubtedly more than 1 character in any test image

    area_thr = lambda img, x: [x for x in img if x > min_area and x < max_area]
    boxes_thr = [[area_thr(image, None) for image in line if area_thr(image, None) != []] for line in
                 segmented_word_box_areas]
    flat_boxes_thr = list(itertools.chain(*list(itertools.chain(*boxes_thr))))

    # Filter artifacts that are still small, but large enough to be mistaken for words, where filtering is based on
    # the average word size of the current document
    outlier_thr = calc_outlier(flat_boxes_thr)
    filtered_word_box_images = []
    character_widths = []
    for i, line in enumerate(segmented_word_box_images):
        line_list = []
        for j, word in enumerate(line):
            if word != []:
                word_list = []
                for k, character in enumerate(word):
                    if character.size != 0:
                        skip_left_pruning = False
                        skip_right_pruning = False
                        if segmented_word_box_areas[i][j][k] >= outlier_thr:
                            new_cluster = words_in_lines[i][j]
                            boundries = all_box_boundaries[i][j][k]
                            if not is_boundary_included(all_box_boundaries[i][j], boundries):
                                x_min = boundries[1][0]
                                x_max = boundries[1][1]
                                taller_cluster = new_cluster[:, x_min - 1:x_max + 1]
                                if x_min == 0:
                                    skip_left_pruning = True
                                if x_max == new_cluster.shape[1]:
                                    skip_right_pruning = True
                                if taller_cluster != []:
                                    if is_image_border_active(taller_cluster):
                                        word_list.append(trim_360(remove_character_artifacts(character, skip_left_pruning,
                                                                                    skip_right_pruning)))
                                    else:
                                        word_list.append(trim_360(remove_character_artifacts(taller_cluster, skip_left_pruning,
                                                                                    skip_right_pruning)))
                                    character_widths.append(word_list[-1].shape[1])
                if word_list != []:
                    line_list.append(word_list)
        if line_list != []:
            filtered_word_box_images.append(line_list)

    return filtered_word_box_images, character_widths


def remove_character_artifacts(image, skip_left_pruning=False, skip_right_pruning=False, min_cluster = 500, internal_min_cluster = 30):
    img_copy = copy.deepcopy(image)
    num_labels, labels = cv2.connectedComponents(img_copy)
    clusters = getComponentClusters(num_labels, labels)

    left_border = img_copy[:, 0]
    right_border = img_copy[:, -1]
    for cluster in clusters:
        if np.sum(cluster) < min_cluster:
            for y, x in cluster:
                if (x == 0 and left_border[y] and not skip_left_pruning
                    or x == img_copy.shape[1] - 1 and right_border[y] and not skip_right_pruning) \
                        and img_copy[y, x]:
                    for y, x in cluster:
                        img_copy[y, x] = 0
                    break
            if np.sum(cluster) < internal_min_cluster:
                for y, x in cluster:
                    for y, x in cluster:
                        img_copy[y, x] = 0
                    break
    #plotSimpleImages([img_copy, image])
    return img_copy

def destructure_characters(characters_in_line):
    characters = []
    for line in characters_in_line:
        for word in line:
            for character in word:
                characters.append(character.astype(int))

    return characters


        
