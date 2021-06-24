import sys
# sys.path.append('C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog')
# sys.path.append('C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/Text_Segmentation/plotting.py')
from Style_Classification.feature_detection import *
import numpy as np
from Text_Segmentation.plotting import plot_simple_images
import torch
import os
import copy
from Text_Segmentation.lineSegmentation import line_segmentation
from Text_Segmentation.wordSegmentation import word_segmentation, trim_360
from Text_Segmentation.characterSegmentation import character_segmentation, remove_character_artifacts, slide_over_word, select_slides, clean_image
from Text_Segmentation.segmentation_to_recog import get_label_probability, TheRecognizer

# image_num = 15
# image_name = "archaic3.jpg"
image_name = 15
dev_path = f"data/image-data/binaryRenamed/{image_name}.jpg"  # development path
# dev_path = f"data/cropped_labeled_images/{image_name}"  # development path
# new_folder_path = f"data/cropped_labeled_images/paths/{image_name[0:-4]}"
new_folder_path = f"data/image-data/binaryRenamed/paths/{str(image_name)}"


# periods_path = "../data/full_images_periods/Hasmonean/hasmonean-330-1.jpg"
# new_folder_path = f"../data/full_images_periods/Hasmonean/paths/{os.path.basename(periods_path).split('.')[0]}"

section_images = line_segmentation(dev_path, new_folder_path)
plot_simple_images(section_images)
lines, words_in_lines = word_segmentation(section_images)
characters, single_character_widths, mean_character_width = character_segmentation(words_in_lines)

model = TheRecognizer()
model.load_model(model.load_checkpoint('40_char_rec.ckpt', map_location=torch.device('cpu')))
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