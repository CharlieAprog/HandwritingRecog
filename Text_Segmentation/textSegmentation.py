import sys
sys.path.append('C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog')
sys.path.append('C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/Text_Segmentation/plotting.py')
from Style_Classification.feature_detection import *
import numpy as np
from plotting import plot_simple_images
import torch
import os
import copy
from Text_Segmentation.lineSegmentation import line_segmentation
from Text_Segmentation.wordSegmentation import word_segmentation, trim_360
from Text_Segmentation.characterSegmentation import character_segmentation, remove_character_artifacts, slide_over_word
from Text_Segmentation.segmentation_to_recog import get_label_probability, TheRecognizer


def clean_image(image, thresh_side=500, thresh_mid=30, trim_thresh=10):
    # image = get_binary(image)
    image = image.astype(np.uint8)
    new = remove_character_artifacts(image, min_cluster=thresh_side, internal_min_cluster=thresh_mid)
    if new.size == 0:
        new = image
    new = trim_360(new, section_thresh=trim_thresh)
    return new


def select_slides(sliding_characters, predicted_char_num, model, window_size, name2idx):
    shift = 1
    chosen_characters = 2

    first = trim_360(sliding_characters[0])
    first_label, _ = get_label_probability(first, model)
    last = trim_360(sliding_characters[-1])
    last_label, _ = get_label_probability(last, model)

    recognised_characters = [first]
    labels = [first_label]
    print(window_size)
    prev_letter_start = 0
    start_idx = 0
    while chosen_characters < predicted_char_num:
        best_prob = 0
        chosen_slide = 0
        chosen_label = 0
        for idx, slide in enumerate(sliding_characters[start_idx:]):
            start = shift * idx
            end = start + window_size
            begin_limit = int(prev_letter_start + window_size * 0.75)
            end_limit = int(prev_letter_start + window_size * 0.75 + window_size + window_size * 0.6)
            # print(begin_limit, end_limit)
            if start >= begin_limit and end <= end_limit:
                trimmed_slide = trim_360(slide)
                predicted_label, probability = get_label_probability(trimmed_slide, model)
                predicted_letter = list(name2idx.keys())[predicted_label]
                print(f'Predicted label:{predicted_letter} probabilty:{probability}')
                print(f"window: [{shift * idx}-{window_size + shift * idx}]")
                if probability > best_prob:
                    best_prob = probability
                    chosen_slide = trimmed_slide
                    chosen_label = predicted_label
                    temp_idx = idx
        chosen_characters += 1
        print('letter chosen')
        start_idx = temp_idx
        recognised_characters.append(clean_image(chosen_slide))
        labels.append(chosen_label)
        prev_letter_start = 0

    recognised_characters.append(last)
    labels.append(last_label)
    return recognised_characters, labels


# image_num = 10
image_name = "archaic1.jpg"
dev_path = f"C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/data/cropped_labeled_images/{image_name}"  # development path
new_folder_path = f"C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/data/cropped_labeled_images/paths/{image_name[0:-4]}"

# periods_path = "../data/full_images_periods/Hasmonean/hasmonean-330-1.jpg"
# new_folder_path = f"../data/full_images_periods/Hasmonean/paths/{os.path.basename(periods_path).split('.')[0]}"

section_images = line_segmentation(dev_path, new_folder_path)
lines, words_in_lines = word_segmentation(section_images)
characters, single_character_widths, mean_character_width = character_segmentation(words_in_lines)

model = TheRecognizer()
model.load_model(model.load_checkpoint('C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/40_char_rec.ckpt', map_location=torch.device('cpu')))
name2idx = {'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Dalet': 3, 'Gimel': 4, 'He': 5,
            'Het': 6, 'Kaf': 7, 'Kaf-final': 8, 'Lamed': 9, 'Mem': 10,
            'Mem-medial': 11, 'Nun-final': 12, 'Nun-medial': 13, 'Pe': 14,
            'Pe-final': 15, 'Qof': 16, 'Resh': 17, 'Samekh': 18, 'Shin': 19,
            'Taw': 20, 'Tet': 21, 'Tsadi-final': 22, 'Tsadi-medial': 23,
            'Waw': 24, 'Yod': 25, 'Zayin': 26}
window_size = int(mean_character_width)
shift = 1
all_segmented_characters = []
all_segmented_labels = []
characters_skipped = 0
for char_idx, character_segment in enumerate(characters):
    if character_segment.shape[1] > mean_character_width + np.std(
            single_character_widths):  # multiple characters suspected
        print("\nMultiple characters classifictiaon")
        predicted_char_num = round(character_segment.shape[1] / mean_character_width)
        sliding_characters = slide_over_word(character_segment, window_size, shift)
        recognised_characters, predicted_labels = select_slides(sliding_characters, predicted_char_num, model,
                                                                window_size, name2idx)
        multiple_characters = copy.deepcopy(recognised_characters)
        multiple_characters.append(character_segment)
        # recognised_characters.append(character_segment)
        predictions_string = ''
        for label in predicted_labels:
            predictions_string = f'{predictions_string}, {list(name2idx.keys())[label]}'
        # plot_simple_images(multiple_characters, title=predictions_string)
        all_segmented_characters.extend(recognised_characters)
        all_segmented_labels.extend(predicted_labels)
    else:  # single character
        print("\nSingle character classification")
        if character_segment.size != 0:
        # try:
        #     character_segment = clean_image(character_segment, thresh_side=50000)
        # except:
        #     plt.imshow(character_segment)
        #     plt.title("get_component_clusters failed.")
        #     plt.show()
        #     exit()

            predicted_label, probability = get_label_probability(character_segment, model)
            predicted_letter = list(name2idx.keys())[predicted_label]
            print(f'Predicted label:{predicted_letter} probabilty:{probability}')
            # plot_simple_images([character_segment], title=f'{predicted_label + 1}:{predicted_letter}')
            all_segmented_characters.append(character_segment)
            all_segmented_labels.append(int(predicted_label))
        else:
            characters_skipped += 1
all_segmented_labels = np.asarray(all_segmented_labels)

print("*"*40)
print("TOTAL CHARACTERS SKIPPED:", characters_skipped)
print("*"*40)
print('Getting style classification for all chararcters:')
style_vec,chi_squared_vec = get_style_char_vec(all_segmented_characters,all_segmented_labels)

n_neighbors = len(style_vec)
dominant_style = get_dominant_style(style_vec,chi_squared_vec,n_neighbors)
print(dominant_style)
    # print(f"window: [{shift*idx}-{window_size + shift*idx}]")