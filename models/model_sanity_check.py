from Text_Segmentation.segmentation_to_recog import *

# two dicts one from label names to idx
name2idx = {'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Dalet': 3, 'Gimel': 4, 'He': 5,
            'Het': 6, 'Kaf': 7, 'Kaf-final': 8, 'Lamed': 9, 'Mem': 10,
            'Mem-medial': 11, 'Nun-final': 12, 'Nun-medial': 13, 'Pe': 14,
            'Pe-final': 15, 'Qof': 16, 'Resh': 17, 'Samekh': 18, 'Shin': 19,
            'Taw': 20, 'Tet': 21, 'Tsadi-final': 22, 'Tsadi-medial': 23,
            'Waw': 24, 'Yod': 25, 'Zayin': 26}

# and one reversed from idx to name
idx2name = {v: k for k, v in name2idx.items()}

# loop over the val data and call the function plus some other stuff I tried
path_to_val_data = '/home/jan/PycharmProjects/HandwritingRecog/data/Char_Recog/final_char_data/val_no_morph/*/*.png'
val_data = glob.glob(path_to_val_data)
cor = 0
wro = 0

class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x < self.thr).to(x.dtype)  # do not change the data type

bin_transform = transforms.Compose([
    transforms.ToTensor(),
    ThresholdTransform(thr_255=250)
])

model = TheRecognizer()
model.load_model(model.load_checkpoint('40_char_rec.ckpt', map_location=torch.device('cpu')))

for img_name in val_data:
    folders = img_name.split('/')
    label = folders[9]
    label_idx = name2idx[label]
    _, img = boundingboxcrop(img_name)
    
    if img != []:
        img = bin_transform(img).numpy()
        pred_label, prob = get_label_probability(img[0], model)

        if pred_label == label_idx:
            cor += 1
        else:
            wro += 1

print(cor/(cor+wro))
