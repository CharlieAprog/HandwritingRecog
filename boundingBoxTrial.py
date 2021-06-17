from matplotlib.pyplot import plot
import itertools
import copy
from Text_Segmentation.plotting import plotSimpleImages, plotGrid
from Text_Segmentation.lineSegmentation import calc_outlier, timer
from Text_Segmentation.plotting import plotSimpleImages
from Text_Segmentation.characterSegmentation import character_segment, get_sliding_words, slide_over_word, getComponentClusters, filter_characters, run_character_segment
from Text_Segmentation.textSegmentation import text_segment
from Text_Segmentation import *
from segmentation_to_recog import *

image_num = 18

#lines = the images of each line of an image
#words_in_lines = each word in each line of image,
#sliding_words = sliding window of each of these words
#TODO   DONE -a make bounding boxes have top and bottom of line instead of max and min of conected component 
#       DONE -b remove any recognised bounding boxes that are inside of a larger bonding box 
#       -b.1 remove artifcats if they touch the boudries
#       -b.2 trim characters
#       -c obtain remaining images of suspected instances where there are multiple characters
#       -d on these suspected multi characters, run sliding window and obtain n images
#       -e run CNN on these n images and predict the label and mark the highest certainty region
#       -f (hopefully do not have to implement ngrams)
#       -g save the regions of the highest certainty predictions to the words in line function
#       -h create list of predictions of all resulting characters and pass for each character the character and the predicted label to style recogniser
#       -i --> recognise style

# |-----------------------------------------------------|
# |              CHARACTER FILTERING                    |
# |-----------------------------------------------------|
lines, words_in_lines = text_segment(image_num)
# Get all characters from all words
segmented_word_box_images, segmented_word_box_areas, all_characters, character_widths, all_box_boundaries = run_character_segment(words_in_lines)
filtered_word_box_images_all_lines = filter_characters(segmented_word_box_areas, segmented_word_box_images, all_box_boundaries, words_in_lines)


# identify long characters
# mean_character_width = np.mean(character_widths)
# for char_idx in range(len(characters)):
#     if character_widths[char_idx] > mean_character_width + np.std(character_widths):
#         print(characters[char_idx])
#         plotSimpleImages([characters[char_idx]])


for i in range(3):
    print(f"Line {i}")
    for z in range(len(segmented_word_box_images[i])):
        plotSimpleImages(segmented_word_box_images[i][z], title="OLD")
    for j in range(len(filtered_word_box_images_all_lines[i])):
        plotSimpleImages(filtered_word_box_images_all_lines[i][j], title="NEW")


# |-----------------------------------------------------|
# |              CHARACTER RECOGNITION                  |
# |-----------------------------------------------------|
window_size = 100
shift = 1
sliding_words = get_sliding_words(words_in_lines, window_size, shift)
# label, prob_bounding_box = get_label_probability(words_in_lines[0][1], model)
# print(label, prob_bounding_box)

# for i, line in enumerate(sliding_words):
#     for j, word in enumerate(line):
#         for slide in word:
#             slide = slide.astype(np.uint8)
#             label, prob_bounding_box = get_label_probability(slide, model)
#             print(label, prob_bounding_box)
#         plotSimpleImages(word)
exit()


lines, words_in_lines = text_segment(image_num) 
# sliding_words = get_sliding_words(words_in_lines,window_size, shift)
# plotSimpleImages(lines)
# character_widths = []
# characters = []
# for line_idx in range(len(words_in_lines)):
#     for word_idx in range(len(words_in_lines[line_idx])):
#         word_image = words_in_lines[line_idx][word_idx]
#         word_box_images, new_word_image = character_segment(word_image) # list of numpy array (images)
#         words_in_lines[line_idx][word_idx] = new_word_image
#         for character in word_box_images:
#             if character.size > 0:
#                 characters.append(character)
#                 character_widths.append(len(character[0,:]))
# mean_character_width = np.mean(character_widths)


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

