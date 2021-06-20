from tarfile import NUL
from matplotlib.pyplot import plot, title
import itertools
import copy
from Text_Segmentation.plotting import plotSimpleImages, plotGrid
from Text_Segmentation.lineSegmentation import calc_outlier, timer, get_binary
from Text_Segmentation.plotting import plotSimpleImages
from Text_Segmentation.characterSegmentation import *
from Text_Segmentation.textSegmentation import text_segment, trim_360
from Text_Segmentation import *
from segmentation_to_recog import *

image_num = 12
def clean_image(image, thresh_side=500, thresh_mid=30, trim_thresh= 10):
    # image = get_binary(image)
    image = image.astype(np.uint8)
    new = remove_character_artifacts(image, min_cluster= thresh_side, internal_min_cluster=thresh_mid)
    if new.size == 0:
        new = image
    new = trim_360(new, line_thresh=trim_thresh)
    return new

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
shift = 1


def select_slides(slides, predicted_char_num, model, window_size):
    shift = 1  
    chosen_characters = 2 

    first = trim_360(slides[0])
    first_label,_ = get_label_probability(first, model)
    last = trim_360(slides[-1])
    last_label,_ = get_label_probability(last, model)

    recognised_characters = [first]
    labels = [first_label]
    print(window_size )
    prev_letter_start = 0
    start_idx= 0
    while chosen_characters < predicted_char_num:
        temp_letter_start = 0
        best_prob = 0
        chosen_slide = 0
        chosen_label = 0
        for idx, slide in enumerate(sliding_characters[start_idx:]):
            start = shift*idx
            end = start + window_size
            begin_limit = int(prev_letter_start + window_size * 0.75)
            end_limit = int(prev_letter_start + window_size * 0.75 + window_size+ window_size * 0.6)
            # print(begin_limit, end_limit)
            if start >= begin_limit and end <= end_limit:
                trimmed_slide = trim_360(slide)
                predicted_label, probability = get_label_probability(trimmed_slide, model)
                predicted_letter = list(name2idx.keys())[predicted_label]
                print(f'Predicted label:{predicted_letter} probabilty:{probability}')
                print(f"window: [{shift*idx}-{window_size + shift*idx}]")
                if probability > best_prob:
                    best_prob = probability
                    chosen_slide = trimmed_slide
                    chosen_label = predicted_label
                    temp_letter_start = start
                    temp_idx = idx
        chosen_characters += 1
        print('letter chosen')
        prev_letter_start = temp_letter_start
        start_idx = temp_idx
        recognised_characters.append(clean_image(chosen_slide))
        labels.append(chosen_label)
        prev_letter_start = 0
    
    recognised_characters.append(last)
    labels.append(last_label)
    return recognised_characters, labels

all_suspected = 0
changed = 0
suspect_indices = []
changed_indices = []
for char_idx, character_segment in enumerate(characters): 
    if character_segment.shape[1] > mean_character_width + np.std(character_widths):
        all_suspected  += 1
        suspect_indices.append(char_idx)
        # Run connected components to get number of labels, so merged clusters are identified beforehand
        character_segment = character_segment.astype(np.uint8)
        num_labels, clusters = cv2.connectedComponents(character_segment, connectivity=4)
        # plotSimpleImages([character_segment], title="Original character")
        clusters = getComponentClusters(num_labels, clusters)
        box_boundaries = getBoundingBoxBoundaries(character_segment, clusters)
        # print('number of bounding boxes:', len(box_boundaries))
        # plotConnectedComponentBoundingBoxes(character_segment, box_boundaries)

        eroded_img_boundaries, eroded_img = erode_clusters(character_segment, kernel=(2,2), iter_num=3)
        print(num_labels, len(eroded_img_boundaries))
        plotSimpleImages([eroded_img], title="eroded character")
        eroded_img_list, _ = get_box_images(eroded_img_boundaries, eroded_img)
        
        
        if len(eroded_img_boundaries) > len(box_boundaries):
            changed += 1
            changed_indices.append(char_idx)
            temp_list = []
            for img in eroded_img_list:
                kernel = np.ones((2,2), np.uint8)
                img = img.astype(np.uint8)
                if img.size > 0:
                    dialated_img = cv2.dilate(img, kernel, iterations=3)
                    temp_list.append(dialated_img)
            temp_list.append(character_segment)
            plotSimpleImages(temp_list, title='Dialation and erosion')

for idx in suspect_indices:
    if idx not in changed_indices:
        plotSimpleImages([characters[idx]], title="Characters failed to erode")

print(f'all:{all_suspected}, changed:{changed}')
for char_idx, character_segment in enumerate(characters): 
    if character_segment.shape[1] > mean_character_width + np.std(character_widths): #multiple characters suspected
        print("\nMultiple characters classifictiaon")
        predicted_char_num = round(character_segment.shape[1]/mean_character_width)
        predicted_char_num_string = f'predicted number of characters:{predicted_char_num}'
        sliding_characters = slide_over_word(character_segment, window_size, shift)
        recognised_characters, predicted_labels = select_slides(sliding_characters, predicted_char_num, model, window_size)
        recognised_characters.append(character_segment)
        predictions_string = ''
        for label in predicted_labels:
            predictions_string = f'{predictions_string}, {list(name2idx.keys())[label]}'
        plotSimpleImages(recognised_characters, title=predictions_string)
        
    else: # single character
        print("\nSingle character classification")
        character_segment = clean_image(character_segment, thresh_side=50000)
        predicted_label, probability = get_label_probability(character_segment, model)
        predicted_letter = list(name2idx.keys())[predicted_label]
        print(f'Predicted label:{predicted_letter} probabilty:{probability}')
        plotSimpleImages([character_segment], title=f'{predicted_label+1}:{predicted_letter}')

            






    # predicted_letter = list(name2idx.keys())[predicted_label]
    # print(f'Predicted label:{predicted_letter} probabilty:{probability}')
    # print(f"window: [{shift*idx}-{window_size + shift*idx}]")


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

