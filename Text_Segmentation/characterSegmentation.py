from Text_Segmentation.plotting import *


def slide_over_word(word, window_size, shift):
    images = []
    height, width = word.shape
    for snap in range(0, width - window_size, shift):
        images.append(word[:, snap:snap + window_size])
        #plotSimpleImages([word[:, snap:snap + window_size]])
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
    kernel = np.ones((4,2), np.uint8)
    word = cv2.dilate(word, kernel, iterations=1)
    num_labels, labels = cv2.connectedComponents(word)
    clusters = getComponentClusters(num_labels, labels)
    box_boundaries = getBoundingBoxBoundaries(word, clusters)
    
    #plotConnectedComponentBoundingBoxes(word, box_boundaries)
    return box_boundaries, word
        
            
def get_box_images(box_boundaries, word):
    box_images = []
    for box in box_boundaries:
        y_min = box[0][0]
        y_max = box[0][1]
        x_min = box[1][0]
        x_max = box[1][1]
        box_img = word[y_min:y_max,x_min:x_max]
        box_images.append(box_img)
    return box_images

def character_segment(word):
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
    box_images = get_box_images(box_boundaries, word)
    # plotConnectedComponentBoundingBoxes(word, box_boundaries)
    print("Character segmentation complete.")
    return box_images, word