from matplotlib.pyplot import plot
import itertools
import copy
from Text_Segmentation.plotting import plotSimpleImages, plotGrid
from Text_Segmentation.lineSegmentation import calc_outlier
from Text_Segmentation.characterSegmentation import character_segment, get_sliding_words
from Text_Segmentation.textSegmentation import text_segment
from Text_Segmentation import *
from segmentation_to_recog import *

image_num = 14
window_size = 70
shift = 20
#lines = the images of each line of an image
#words_in_lines = each word in each line of image,
#sliding_words = sliding window of each of these words
lines, words_in_lines = text_segment(image_num)
# sliding_words = get_sliding_words(words_in_lines, window_size, shift)

# Get all characters from all words
segmented_word_box_images = []
segmented_word_box_areas = []
for line in words_in_lines:
    line_word_images = []
    line_word_areas = []
    for word in line:
        pixels, areas = character_segment(word, title="[OLD]")
        line_word_images.append(pixels)
        line_word_areas.append(areas)
    segmented_word_box_images.append(line_word_images)
    segmented_word_box_areas.append(line_word_areas)

# Filter nonsense artifacts
area_thr = lambda img, x: [x for x in img if x > 500]
boxes_thr = [[area_thr(image, len(image)) for image in line if area_thr(image, len(image)) != []] for line in segmented_word_box_areas]
flat_boxes_thr = list(itertools.chain(*list(itertools.chain(*boxes_thr))))

# Filter artifacts that are still small, but large enough to be mistaken for words, where filtering is based on
# the average word size of the current document
outlier_thr = calc_outlier(flat_boxes_thr)
filtered_word_box_images_all_lines = [
    [
        [cluster for k, cluster in enumerate(pixel_clusters) if segmented_word_box_areas[i][j][k] >= outlier_thr]
        for j, pixel_clusters in enumerate(line) if pixel_clusters != []
    ]
    for i, line in enumerate(segmented_word_box_images)
]
filtered_word_box_images_all_lines = [[word[0] for word in line if word != []] for line in filtered_word_box_images_all_lines]

# TODO: Whether 'filtered_word_box_images_all_lines' contains the proper amount of images or not is unknown

sliding_words = get_sliding_words(filtered_word_box_images_all_lines, window_size, shift)
label, prob_bounding_box = get_label_probability(filtered_word_box_images_all_lines[0][1], model)
print(label, prob_bounding_box)
plotSimpleImages(sliding_words[0][0])

for slide in sliding_words[0][0]:
    slide = slide.astype(np.uint8)
    label, prob_bounding_box = get_label_probability(slide, model)
    print(label, prob_bounding_box)

