from operator import mul
from pathlib import Path
# from font2image import *
from matplotlib.pyplot import plot
# sys.path.append('C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog')
# sys.path.append('C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/Text_Segmentation/plotting.py')
from Style_Classification.Classify_Char_Style import *
import numpy as np
from Text_Segmentation.plotting import plot_simple_images
import torch
import os
import copy
# from font2image import *
from Text_Segmentation.lineSegmentation import line_segmentation
from Text_Segmentation.wordSegmentation import word_segmentation, trim_360
from Text_Segmentation.characterSegmentation import character_segmentation, remove_character_artifacts, slide_over_word, select_slides, clean_image, destructure_characters
from Text_Segmentation.segmentation_to_recog import get_label_probability, TheRecognizer
import glob
import argparse

from operator import mul
from pathlib import Path

from matplotlib.pyplot import plot
# sys.path.append('C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog')
# sys.path.append('C:/Users/Panos/Desktop/HandwritingRecognition/HandwritingRecog/Text_Segmentation/plotting.py')
from Style_Classification.Classify_Char_Style import *
import numpy as np
from Text_Segmentation.plotting import plot_simple_images
import torch
import os
import copy
from Text_Segmentation.lineSegmentation import line_segmentation
from Text_Segmentation.wordSegmentation import word_segmentation, trim_360
from Text_Segmentation.characterSegmentation import character_segmentation, remove_character_artifacts, slide_over_word, select_slides, clean_image, destructure_characters
from Text_Segmentation.segmentation_to_recog import get_label_probability, TheRecognizer

# data/cropped_labeled_images/
parser = argparse.ArgumentParser()
parser.add_argument('path', help= 'path to folder with image files')
args = parser.parse_args()
test_folder = args.path
test_images= glob.glob(test_folder+"*")
print('test_images')
print(test_images)
for image_path in test_images:


    # image_num = 15
    # image_names = ["25-Fg001.pbm", "124-Fg004.pbm", "archaic1.jpg", "archaic2.jpg", "archaic3.jpg",
    #                 "hasmonean3.jpg", "hasmonian1.jpg", "herodian1.jpg", "herodian2.jpg", "herodian3.jpg"]
    image_name = "archaic1.jpg"
    #image_name = 15

    # new images
    # dev_path = f"data/cropped_labeled_images/{image_name}"  # development path
    # dev_path = f"data/cropped_labeled_images/{image_name}"  # development path
    # new_folder_path = f"data/cropped_labeled_images/paths/{image_name[0:-4]}"
    # new_folder_path = f"data/cropped_labeled_images/paths/{image_name[0:-4]}"

    # binary images
    dev_path = f"data/image-data/binaryRenamed/{3}.jpg"  # development path
    aStar_path = f"data/image-data/binaryRenamed/paths/{3}"

    section_images = line_segmentation(dev_path, aStar_path)
    # plot_simple_images(section_images)
    lines, words_in_lines = word_segmentation(section_images)
    characters_word_line, single_character_widths, mean_character_width = character_segmentation(words_in_lines)

    # plot_simple_images(lines)
    # for line in words_in_lines:
    #     for word in line:
    #         plot_simple_images(word)

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
                    print("\nMultiple characters classifictiaon")
                    predicted_char_num = round(character_segment.shape[1] / mean_character_width)
                    sliding_characters = slide_over_word(character_segment, window_size, shift)
                    recognised_characters, predicted_labels, probabilities= select_slides(sliding_characters, predicted_char_num, model,
                                                                            window_size, name2idx)
                    multiple_characters = copy.deepcopy(recognised_characters)
                    multiple_characters.append(character_segment)
                    # plot_simple_images(multiple_characters)
                    # recognised_characters.append(character_segment)
                    predictions_string = ''
                    for label in predicted_labels:
                        predictions_string = f'{predictions_string}, {list(name2idx.keys())[label]}'
                    plot_simple_images(multiple_characters, title=predictions_string)
                    word_imgs.extend(recognised_characters)
                    word_labels.extend(predicted_labels)
                    # all_segmented_characters.extend(recognised_characters)
                    # all_segmented_labels.extend(predicted_labels)
                    all_char_propabilities.extend(probabilities)
                else:  # single character
                    print("\nSingle character classification")
                    if character_segment.size != 0:
                    # try:
                        character_segment = clean_image(character_segment)
                        # plot_simple_images([character_segment])
                    # except:
                    #     plt.imshow(character_segment)
                    #     plt.title("get_component_clusters failed.")
                    #     plt.show()
                    #     exit()

                        predicted_label, probability = get_label_probability(character_segment, model)
                        predicted_letter = list(name2idx.keys())[predicted_label]
                        print(f'Predicted label:{predicted_letter} probabilty:{probability}')
                        plot_simple_images([character_segment], title=f'{predicted_label + 1}:{predicted_letter}')
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

    for character in destructure_characters(all_segmented_characters):
        plot_simple_images([character], title='hello')

    labels_for_file = copy.deepcopy(all_segmented_labels)


    # work with these for style classification, they are in one array
    all_segmented_labels = destructure_characters(all_segmented_labels)
    all_segmented_characters = destructure_characters(all_segmented_characters)

    all_segmented_labels = np.asarray(all_segmented_labels)
    all_char_propabilities = np.asarray(all_char_propabilities)

    print("*"*40)
    print("TOTAL CHARACTERS SKIPPED:", characters_skipped)
    print("*"*40)
    print("*"*40)
    print("TOTAL NUMBER OF CHARACTERS:", len(all_segmented_characters))
    print("*"*40)
    print("*"*40)
    print('Getting style classification for all chararcters:')
    #get style labels for each character in the image
    style_vec,chi_squared_vec = get_style_char_vec(all_segmented_characters,all_segmented_labels,all_char_propabilities,prob_threshold = 0.7, global_vec=False, show_hinge_points=2)

    #get dominant style of image
    n_neighbors = 10
    dominant_style = get_dominant_style(style_vec,chi_squared_vec,n_neighbors)
    print(dominant_style)
    print(max(dominant_style))


    Path('results/').mkdir(parents=True, exist_ok=True) 

    
        # print(f"window: [{shift*idx}-{window_size + shift*idx}]")w: [{shift*idx}-{window_size + shift*idx}]")