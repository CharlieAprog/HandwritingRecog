from Text_Segmentation.plotting import *
from Text_Segmentation.textSegmentation import calc_outlier
import itertools

def slide_over_word(word, window_size, shift):
    images = []
    height, width = word.shape
    for snap in range(0, width - window_size, shift):
        images.append(word[:, snap:snap + window_size])
        # plotSimpleImages([word[:, snap:snap + window_size]])
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
        box_boundaries.append([[y_max, y_min], [x_min, x_max]])
    return box_boundaries


def dialate_clusters(num_boxes, word):
    kernel = np.ones((5,3), np.uint8)
    word = cv2.dilate(word, kernel, iterations=1)
    num_labels, labels = cv2.connectedComponents(word)
    clusters = getComponentClusters(num_labels, labels)
    box_boundaries = getBoundingBoxBoundaries(word, clusters)
    #plotConnectedComponentBoundingBoxes(word, box_boundaries)
    return box_boundaries, word
        
            
def get_box_images(box_boundaries, word):
    """Returns all the bounded images and areas thereof within a word that can contain any number of characters"""
    box_images = []
    box_areas = []
    for box in box_boundaries:
        y_min = box[0][0]
        y_max = box[0][1]
        x_min = box[1][0]
        x_max = box[1][1]
        box_img = word[y_min:y_max, x_min:x_max]
        box_areas.append(abs(y_max-y_min)*abs(x_max-x_min))
        box_images.append(box_img)
    return box_images, box_areas


def character_segment(word, title=None):
    cluster_threshold = 7
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
    return box_images, box_areas, word


def filter_characters(segmented_word_box_areas, segmented_word_box_images):
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
    filtered_word_box_images = [
        [
            [cluster for k, cluster in enumerate(pixel_clusters) if segmented_word_box_areas[i][j][k] >= outlier_thr]
            for j, pixel_clusters in enumerate(line) if pixel_clusters != []
        ]
        for i, line in enumerate(segmented_word_box_images)
    ]

    # filtered_word_box_images = [[word for word in line if word != []] for line in filtered_word_box_images]

    lines_words_char_images = []
    for line in filtered_word_box_images:
        line_imgs = []
        for word in line:
            if word != []:
                line_imgs.append(word)
        lines_words_char_images.append(line_imgs)

    return lines_words_char_images


def run_character_segment(words_in_lines):
    segmented_word_box_images = []
    segmented_word_box_areas = []
    all_characters = []
    character_widths = []
    for line_idx, line in enumerate(words_in_lines):
        line_word_images = []
        line_word_areas = []
        for word_idx, word in enumerate(line):
            pixels, areas, new_word = character_segment(word, title="[OLD]")
            words_in_lines[line_idx][word_idx] = new_word
            line_word_images.append(pixels)
            line_word_areas.append(areas)
            for character in pixels:
                if character.size > 0:
                    all_characters.append(character)
                    character_widths.append(len(character[0,:]))
        segmented_word_box_images.append(line_word_images)
        segmented_word_box_areas.append(line_word_areas)
    return segmented_word_box_images, segmented_word_box_areas, all_characters, character_widths