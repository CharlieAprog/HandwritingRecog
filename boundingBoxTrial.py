from matplotlib.pyplot import plot
import itertools
import copy
from Text_Segmentation.plotting import plotSimpleImages, plotGrid
from Text_Segmentation.lineSegmentation import calc_outlier
from Text_Segmentation.plotting import plotSimpleImages
from Text_Segmentation.characterSegmentation import character_segment, get_sliding_words, slide_over_word
from Text_Segmentation.textSegmentation import text_segment
from Text_Segmentation import *
from segmentation_to_recog import *

image_num = 18
window_size = 100
shift = 60
#lines = the images of each line of an image
#words_in_lines = each word in each line of image,
#sliding_words = sliding window of each of these words

# |-----------------------------------------------------|
# |              CHARACTER FILTERING                    |
# |-----------------------------------------------------|
# TODO: Whether 'filtered_word_box_images_all_lines' contains the proper amount of images or not is unknown
lines, words_in_lines = text_segment(image_num)
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


# |-----------------------------------------------------|
# |              CHARACTER RECOGNITION                  |
# |-----------------------------------------------------|
sliding_words = get_sliding_words(filtered_word_box_images_all_lines, window_size, shift)
label, prob_bounding_box = get_label_probability(filtered_word_box_images_all_lines[0][1], model)
print(label, prob_bounding_box)
plotSimpleImages(sliding_words[0][0])

for slide in sliding_words[0][0]:
    slide = slide.astype(np.uint8)
    label, prob_bounding_box = get_label_probability(slide, model)


lines, words_in_lines = text_segment(image_num) 
<<<<<<< HEAD
# sliding_words = get_sliding_words(words_in_lines,window_size, shift)
plotSimpleImages(lines)
character_widths = []
characters = []
for line_idx in range(len(words_in_lines)):
    for word_idx in range(len(words_in_lines[line_idx])):
        word_image = words_in_lines[line_idx][word_idx]
        word_box_images, new_word_image = character_segment(word_image) # list of numpy array (images)
        words_in_lines[line_idx][word_idx] = new_word_image
        for character in word_box_images:
            if character.size > 0:
                characters.append(character)
                character_widths.append(len(character[0,:]))
mean_character_width = np.mean(character_widths)

for char_idx in range(len(characters)):
    if character_widths[char_idx] > mean_character_width + np.std(character_widths):
        print(characters[char_idx])
        plotSimpleImages([characters[char_idx]])


# plotSimpleImages(lines)
# plotSimpleImages(sliding_words[2][0])
# plotSimpleImages(sliding_words[2][2])
# print(words_in_lines)
# print(len(words_in_lines[0]))
# print(len(words_in_lines[0][0]))
# for line in range(len(words_in_lines[0])):
#     for word in words_in_lines[0]:
#         plotSimpleImages([word])
#         plotSimpleImages([words_in_lines[0][0]])
#         word_boxes_img = character_segment(word)
#         plotSimpleImages(word_boxes_img)
#word_img = 
#box_img = 
#bounding_box_cords = x, y
# for img in word_box_images:
#     label, prob_bounding_box = get_label_probability(img, model)
#     print(idx2name[int(label)])
#     print(label, prob_bounding_box)
#     dim = img.shape
#     print(dim)
#     images = slide_over_word(img, 30, 10)
#     plotSimpleImages(images)
# plotSimpleImages(word_box_images)
=======
sliding_words = get_sliding_words(words_in_lines,window_size, shift)
word_box_images = character_segment(words_in_lines[1][2]) # list of numpy array (images)
>>>>>>> a51425a80ea0fd30f11842d179de8151f6d6ac2b

all_words = [word for line in words_in_lines for word in line]
print(all_words[0])
for word in all_words:
    plotSimpleImages([word])

# plotSimpleImages(sliding_words[2][2])
#
# for slide in sliding_words[2][2]:
#     slide = slide.astype(np.uint8)
#
#     label, prob_bounding_box = get_label_probability(slide, model)
#     print(label, prob_bounding_box)

