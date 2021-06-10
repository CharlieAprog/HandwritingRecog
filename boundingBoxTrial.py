from matplotlib.pyplot import plot
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
lines, words_in_lines = text_segment(image_num) 
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


# plotSimpleImages(sliding_words[2][2])
#
# for slide in sliding_words[2][2]:
#     slide = slide.astype(np.uint8)
#
#     label, prob_bounding_box = get_label_probability(slide, model)
#     print(label, prob_bounding_box)

