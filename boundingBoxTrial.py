from matplotlib.pyplot import plot, title
import itertools
import copy
from Text_Segmentation.plotting import plotSimpleImages, plotGrid
from Text_Segmentation.lineSegmentation import calc_outlier, timer
from Text_Segmentation.plotting import plotSimpleImages
from Text_Segmentation.characterSegmentation import character_segment, \
    get_sliding_words, slide_over_word, getComponentClusters, filter_characters, run_character_segment, destructure_characters
from Text_Segmentation.textSegmentation import text_segment
from Text_Segmentation import *
from segmentation_to_recog import *

image_num = 18

#lines = the images of each line of an image
#words_in_lines = each word in each line of image,
#sliding_words = sliding window of each of these words
#TODO   
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
segmented_word_box_images, segmented_word_box_areas, all_box_boundaries = run_character_segment(words_in_lines)
filtered_word_box_images_all_lines, character_widths = filter_characters(segmented_word_box_areas, segmented_word_box_images, all_box_boundaries, words_in_lines)



# |-----------------------------------------------------|
# |              CHARACTER RECOGNITION                  |
# |-----------------------------------------------------|
characters = destructure_characters(filtered_word_box_images_all_lines)
mean_character_width = np.mean(character_widths)
model = TheRecognizer()
model.load_model(model.load_checkpoint('40_char_rec.ckpt', map_location=torch.device('cpu')))
print(mean_character_width + np.std(character_widths))


name2idx = {'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Dalet': 3, 'Gimel' : 4, 'He': 5,
                'Het': 6, 'Kaf': 7, 'Kaf-final': 8, 'Lamed': 9, 'Mem': 10,
                'Mem-medial': 11, 'Nun-final': 12, 'Nun-medial': 13, 'Pe': 14,
                'Pe-final': 15, 'Qof': 16, 'Resh': 17, 'Samekh': 18, 'Shin': 19,
                'Taw': 20, 'Tet': 21, 'Tsadi-final': 22, 'Tsadi-medial': 23,
                'Waw': 24, 'Yod': 25, 'Zayin': 26}


prediction_list = []
window_size = int(mean_character_width)
shift = 10
for char_idx, character_segment in enumerate(characters): 
    if character_segment.shape[1] > mean_character_width + np.std(character_widths): #multiple characters suspected
        print("\nMultiple characters classifictiaon")
        sliding_characters = slide_over_word(character_segment, window_size, shift)
        for idx, slide in enumerate(sliding_characters):
            predicted_label, probability = get_label_probability(slide, model)
            predicted_letter = list(name2idx.keys())[predicted_label]
            print(f'Predicted label:{predicted_letter} probabilty:{probability}')
            print(f"window: [{shift*idx}-{window_size + shift*idx}]")
        plotSimpleImages([character_segment], title='identified multi character')
    else: # single character
        print("\nSingle character classification")
        predicted_label, probability = get_label_probability(character_segment, model)
        predicted_letter = list(name2idx.keys())[predicted_label]
        print(f'Predicted label:{predicted_letter} probabilty:{probability}')
        plotSimpleImages([character_segment], title=f'{predicted_label+1}:{predicted_letter}')

            


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

