import numpy as np
import imageio
from imgaug import augmenters as iaa
from numpy import asarray
from PIL import Image
from tqdm import tqdm
import Augmentor
import os



def augment():
    letter_paths = os.listdir('data/Style_classification/Herodian')
    for letter_path in letter_paths:
        print(letter_path)
        p = Augmentor.Pipeline(f'data/Style_classification/Herodian/{letter_path}')
        p.random_distortion(probability=0.5, grid_width=3, grid_height=3, magnitude=5)
        p.gaussian_distortion(probability=0.5, grid_width=3, grid_height=3, magnitude=5, corner='bell', method='in')
        p.shear(0.5, 10, 10)  #Size Preserving Shearing
        p.skew_tilt(0.5, magnitude=0.2) #Perspective Transforms 
        p.sample(50)

augment()