import numpy as np
import imageio
from imgaug import augmenters as iaa
from numpy import asarray
from PIL import Image
from tqdm import tqdm
import Augmentor
import os





def augment(images):
    letter_paths = os.import_dir('data/Style_classification/Archaic')
    for letter_path in letter_paths:
        p = Augmentor.Pipeline('./pathToImages')