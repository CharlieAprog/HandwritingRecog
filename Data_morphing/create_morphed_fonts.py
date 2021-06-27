#Uses pillow (you can also use another imaging library if you want)
import os.path

import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

#Load the font and set the font size to 42
font = ImageFont.truetype('/home/jan/PycharmProjects/HandwritingRecog/data/habbakuk/Habbakuk.TTF', 42)

#Character mapping for each of the 27 tokens
char_map = {'Alef' : ')',
            'Ayin' : '(',
            'Bet' : 'b',
            'Dalet' : 'd',
            'Gimel' : 'g',
            'He' : 'x',
            'Het' : 'h',
            'Kaf' : 'k',
            'Kaf-final' : '\\',
            'Lamed' : 'l',
            'Mem' : '{',
            'Mem-medial' : 'm',
            'Nun-final' : '}',
            'Nun-medial' : 'n',
            'Pe' : 'p',
            'Pe-final' : 'v',
            'Qof' : 'q',
            'Resh' : 'r',
            'Samekh' : 's',
            'Shin' : '$',
            'Taw' : 't',
            'Tet' : '+',
            'Tsadi-final' : 'j',
            'Tsadi-medial' : 'c',
            'Waw' : 'w',
            'Yod' : 'y',
            'Zayin' : 'z'}

#Returns a grayscale image based on specified label of img_size
def create_image(label, img_size):
    if (label not in char_map):
        raise KeyError('Unknown label!')

    #Create blank image and create a draw interface
    img = Image.new('L', img_size, 255)
    draw = ImageDraw.Draw(img)

    #Get size of the font and draw the token in the center of the blank image
    w,h = font.getsize(char_map[label])
    draw.text(((img_size[0]-w)/2, (img_size[1]-h)/2), char_map[label], 0, font)

    return img

#Create a 40x40 image of the Alef token and save it to disk
#To get the raw data cast it to a numpy array
save_path = '/home/jan/PycharmProjects/HandwritingRecog/Data_morphing/example_chars/'
for label in char_map:
    os.mkdir(os.path.join(save_path, label))
    img = create_image(label, (40, 40))
    path = save_path + label + '/' + label + '.png'
    img.save(path)