from operator import mul
from pathlib import Path
from font2image import labeltotext, styletotext
from matplotlib.pyplot import plot
from Style_Classification.Classify_Char_Style import *
from Style_Classification.SVM_Style import get_style_SVM
import numpy as np
from Text_Segmentation.plotting import plot_simple_images
import torch
import os
import copy
from Text_Segmentation.lineSegmentation import line_segmentation
from Text_Segmentation.wordSegmentation import word_segmentation, trim_360
from Text_Segmentation.characterSegmentation import character_segmentation, remove_character_artifacts, slide_over_word, select_slides, clean_image, destructure_characters
from Text_Segmentation.segmentation_to_recog import get_label_probability, TheRecognizer
import glob
import argparse
import warnings

warnings.filterwarnings('ignore')

# data/cropped_labeled_images/
#parsing path for image folder
parser = argparse.ArgumentParser()
parser.add_argument('path', help= 'path to folder with image files')
args = parser.parse_args()

#get input path of folder and separate img names w/ glob
test_folder = args.path
test_images= glob.glob(test_folder+"/*")
test_images = [os.path.basename(image_name) for image_name in test_images]
print('test_images:', test_images)
for img_name in test_images:
    print(f"Running Handwriting Recognition on image {img_name}")
    try:
        img_path = f"{test_folder}/{img_name}"
        aStar_path = f"paths/{img_name[0:-4]}"

        print("Running line segmentation...")
        section_images = line_segmentation(img_path, aStar_path)
        print("Running word segmentation...")
        lines, words_in_lines = word_segmentation(section_images)
        print("Running character segmentation...")
        characters_word_line, single_character_widths, mean_character_width = character_segmentation(words_in_lines)

        model = TheRecognizer()
        model.load_model(model.load_checkpoint('40_char_rec.ckpt', map_location=torch.device('cpu')))
        name2idx = {'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Dalet': 3, 'Gimel': 4, 'He': 5,
                    'Het': 6, 'Kaf': 7, 'Kaf-final': 8, 'Lamed': 9, 'Mem': 10,
                    'Mem-medial': 11, 'Nun-final': 12, 'Nun-medial': 13, 'Pe': 14,
                    'Pe-final': 15, 'Qof': 16, 'Resh': 17, 'Samekh': 18, 'Shin': 19,
                    'Taw': 20, 'Tet': 21, 'Tsadi-final': 22, 'Tsadi-medial': 23,
                    'Waw': 24, 'Yod': 25, 'Zayin': 26}
        window_size = int(mean_character_width) if int(mean_character_width) > 90 else 90
        shift = 1
        all_segmented_characters = []
        all_segmented_labels = []
        all_char_propabilities = []
        characters_skipped = 0
        for line in characters_word_line:
            line_imgs = []
            line_labels = []
            for word in line:
                word_imgs = []
                word_labels = []
                for char_idx, character_segment in enumerate(word):
                    if character_segment.shape[1] > mean_character_width + np.std(
                            single_character_widths):  # multiple characters suspected
                        try:
                            # print("\nMultiple characters classifictiaon")
                            predicted_char_num = round(character_segment.shape[1] / mean_character_width)
                            sliding_characters = slide_over_word(character_segment, window_size, shift)
                            recognised_characters, predicted_labels, probabilities = select_slides(sliding_characters, predicted_char_num, model,
                                                                                    window_size, name2idx)
                            multiple_characters = copy.deepcopy(recognised_characters)
                            multiple_characters.append(character_segment)
                            # plot_simple_images(multiple_characters)
                            # recognised_characters.append(character_segment)
                            predictions_string = ''
                            for label in predicted_labels:
                                predictions_string = f'{predictions_string}, {list(name2idx.keys())[label]}'
                            # plot_simple_images(multiple_characters, title=predictions_string)
                            word_imgs.extend(recognised_characters)
                            word_labels.extend(predicted_labels)
                            # all_segmented_characters.extend(recognised_characters)
                            # all_segmented_labels.extend(predicted_labels)
                            all_char_propabilities.extend(probabilities)
                        except:
                            continue

                    else:  # single character
                        # print("\nSingle character classification")
                        if character_segment.size != 0:
                            character_segment = clean_image(character_segment)
                            predicted_label, probability = get_label_probability(character_segment, model)
                            predicted_letter = list(name2idx.keys())[predicted_label]
                            # print(f'Predicted label:{predicted_letter} probabilty:{probability}')
                            #plot_simple_images([character_segment], title=f'{predicted_label + 1}:{predicted_letter}')
                            word_imgs.append(character_segment)
                            word_labels.append(predicted_label)
                            # all_segmented_characters.append(character_segment)
                            # all_segmented_labels.append(int(predicted_label))
                            all_char_propabilities.append(probability)
                        else:
                            characters_skipped += 1
                line_imgs.append(word_imgs[::-1])
                line_labels.append(word_labels[::-1])
            all_segmented_characters.append(line_imgs[::-1])
            all_segmented_labels.append(line_labels[::-1])

        # for character in destructure_characters(all_segmented_characters):
        #     plot_simple_images([character], title='hello')

        labels_for_file = copy.deepcopy(all_segmented_labels)


        # work with these for style classification, they are in one array
        all_segmented_labels = destructure_characters(all_segmented_labels)
        all_segmented_characters = destructure_characters(all_segmented_characters)

        all_segmented_labels = np.asarray(all_segmented_labels)
        all_char_propabilities = np.asarray(all_char_propabilities)

        print("*"*40)
        print("TOTAL NUMBER OF CHARACTERS:", len(all_segmented_characters))
        print("*"*40)
        print('Getting style classification for all chararcters:')


        #get style labels for each character in the image
        dominant_style = get_style_SVM(all_segmented_characters, all_segmented_labels, all_segmented_labels)


        Path('results/').mkdir(parents=True, exist_ok=True) 
        labeltotext(labels_for_file,img_name[0:-4])
        styletotext(dominant_style,img_name[0:-4])
        print(f"Image {img_name} has been processed successfully.\n")
    except Exception as error:
        print("Fatal Error:", error)
        print(f"Skipping image [{img_name}]")
