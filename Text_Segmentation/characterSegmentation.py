from plotting import *


def slide_over_word(word, window_size, shift):
    images = []
    height, width = word.shape
    for snap in range(0, width - window_size, shift):
        images.append(word[:, snap:snap + window_size])
        #plotSimpleImages([word[:, snap:snap + window_size]])
    return images


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