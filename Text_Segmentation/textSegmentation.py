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
from Text_Segmentation.characterSegmentation import character_segmentation, remove_character_artifacts, slide_over_word
from Text_Segmentation.segmentation_to_recog import get_label_probability, TheRecognizer





