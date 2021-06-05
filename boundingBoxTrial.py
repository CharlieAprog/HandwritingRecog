from matplotlib.pyplot import plot
from Text_Segmentation.plotting import plotSimpleImages
from Text_Segmentation.characterSegmentation import character_segment, get_sliding_words
from Text_Segmentation.textSegmentation import text_segment
from Text_Segmentation import *
from segmentation_to_recog import *

image_num = 8
window_size = 70
shift = 20
#lines = the images of each line of an image
#words_in_lines = each word in each line of image,
#sliding_words = sliding window of each of these words
lines, words_in_lines = text_segment(image_num) 
sliding_words = get_sliding_words(words_in_lines,window_size, shift)
word_box_images = character_segment(words_in_lines[2][2]) # list of numpy array (images)
# plotSimpleImages(word_box_images)
# plotSimpleImages(sliding_words[2][2])



#word_img = 
#box_img = 
#bounding_box_cords = x, y
label, prob_bounding_box = get_label_probability(word_box_images[0], model)
print(label, prob_bounding_box)
plotSimpleImages(sliding_words[2][2])

for slide in sliding_words[2][2]:
    slide = slide.astype(np.uint8)
    
    label, prob_bounding_box = get_label_probability(slide, model)
    print(label, prob_bounding_box)

