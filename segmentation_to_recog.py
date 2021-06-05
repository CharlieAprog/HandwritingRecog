import cv2
import torch
from torch import nn
import glob
from Character_Recog.data_investigation import *
import seaborn as sn
from sklearn.metrics import confusion_matrix

class TheRecognizer(nn.Module):
  def __init__(self):
    super(TheRecognizer, self).__init__()
    self.conv_layers = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5,stride=1, padding=0),
        nn.BatchNorm2d(10),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(10, 15, 5, 1, 0),
        nn.BatchNorm2d(15),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    self.lin_layers = nn.Sequential(
        nn.Linear(7*7*15, 300),
        nn.ReLU(),
        nn.Linear(300, 27),
        nn.LogSoftmax(dim=1)
    )
    self.opt = torch.optim.Adam(params=self.parameters(), weight_decay=0.002)

  def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 7*7*15)
        x = self.lin_layers(x)

        return x
  def load_checkpoint(self, ckpt_path, map_location=None):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
        return ckpt

  def save_checkpoint(self, state, save_path):
        torch.save(state, save_path)

  def load_model(self, ckpt):
        self.epoch = ckpt['epoch']
        self.load_state_dict(ckpt['weights'])
        self.opt.load_state_dict(ckpt['optimizer'])

# input: image, new_height, new_width
# output: the same image but resized, padded with
# no distortions
def resize_pad(img, height_all, width_all):
    height, width = img.shape
    if height > width:
        resize_factor = height_all / height
        new_width = int(img.shape[1] * resize_factor)
        dim = (new_width, height_all)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        left_pad = int((width_all - new_width) / 2)
        right_pad = left_pad
        if left_pad + right_pad + new_width != 40:
            right_pad += 1
        resized_pad_img = cv2.copyMakeBorder(resized_img, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=255)
    else:
        resize_factor = height_all / height
        new_height = int(img.shape[0] * resize_factor)
        dim = (width_all, new_height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        top_pad = int((height_all - new_height) / 2)
        bottom_pad = top_pad
        if top_pad + bottom_pad + new_height != 40:
            bottom_pad += 1
        resized_pad_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT,value = 255)

    return resized_pad_img

# given the model and an images this resizes the image to (40, 40)
# feeds it to the model and returns the label_idx and the probability of the label_idx
# could use the probability to tell if an img is a char I guess
def get_label_probability(img, model):
    img = resize_pad(img, 40, 40)
    # make img ready for model
    img_res = np.reshape(img, [1, 1, 40, 40])
    img_torch = torch.Tensor(img_res)
    out = model(img_torch)
    pred_label = torch.argmax(out).detach().numpy()
    out = out.detach().numpy()
    # output of model is log softmax -> this makes it probability distribution
    char_probs = np.exp(out) / (np.exp(out)).sum()

    return pred_label, char_probs[0][pred_label]
    # some way of telling if an img is a char I tried but doesnt work
    # commented for now
    # max_idx = char_probs[0].argsort()[-4:][::-1]
    # # now check if the max_3 probs are close to each other
    # # distance between 2nd highest and highest prob
    # dist1 = abs(char_probs[0][max_idx[1]] - char_probs[0][max_idx[0]])
    # # distance between 2nd highest and 3rd highest prob
    # dist2 = abs(char_probs[0][max_idx[1]] - char_probs[0][max_idx[2]])
    # # distance between 2nd highest and 3rd highest prob
    # dist3 = abs(char_probs[0][max_idx[2]] - char_probs[0][max_idx[3]])
    # if dist1 < 0.3 and dist2 < 0.2 and dist3 < 0.1:
    #     print(char_probs[0][max_idx[0]])
    #     print(dist1, dist2)
    #     return False
    # else:
    #     return True
# two dicts one from label names to idx
name2idx = {'Alef': 0, 'Ayin': 1, 'Bet': 2, 'Dalet': 3, 'Gimel' : 4, 'He': 5,
            'Het': 6, 'Kaf': 7, 'Kaf-final': 8, 'Lamed': 9, 'Mem': 10,
            'Mem-medial': 11, 'Nun-final': 12, 'Nun-medial': 13, 'Pe': 14,
            'Pe-final': 15, 'Qof': 16, 'Resh': 17, 'Samekh': 18, 'Shin': 19,
            'Taw': 20, 'Tet': 21, 'Tsadi-final': 22, 'Tsadi-medial': 23,
            'Waw': 24, 'Yod': 25, 'Zayin': 26}
# and one reversed from idx to name
idx2name = {v: k for k, v in name2idx.items()}

model = TheRecognizer()
model.load_model(model.load_checkpoint('40_char_rec.ckpt', map_location=torch.device('cpu')))




# loop over the val data and call the function plus some other stuff I tried
# path_to_val_data = '/home/jan/PycharmProjects/HandwritingRecog/data/Char_Recog/binarized_monkbrill_split_40x40_morphed/val_no_morph/*/*png'
# val_data = glob.glob(path_to_val_data)
# for img_name in val_data:
#     folders = img_name.split('/')
#     label = folders[9]
#     label_idx = name2idx[label]
#     _, img = boundingboxcrop(img_name)
#     label, prob = get_label_probability(img, model)
#     print(label, prob)

    # img = resize_pad(img, 40, 40)
    # img_res = np.reshape(img, [1, 1, 40, 40])
    # img_torch = torch.Tensor(img_res)
    # out = model(img_torch)
    # pred_label = torch.argmax(out)
    # y_true.append(label_idx)
    # y_pred.append(pred_label)
    # out = out.detach().numpy()
    # char_probs = np.exp(out) / (np.exp(out)).sum()
    # if pred_label != label_idx:
    #     wrong += 1
    #
    #     print(char_probs[0][pred_label])
    #     if char_probs[0][pred_label] < 0.6:
    #         x = int(pred_label.numpy())
    # #         print(label, idx2name[x])
    # else:
    #     correct += 1
# c_mat = confusion_matrix(y_true, y_pred)
# sn.heatmap(c_mat)
# print(c_mat)
# plt.show()






