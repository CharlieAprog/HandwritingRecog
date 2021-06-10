from segmentation_to_recog import *


# loop over the val data and call the function plus some other stuff I tried
path_to_val_data = '/home/jan/PycharmProjects/HandwritingRecog/data/Char_Recog/binarized_monkbrill_split_40x40_morphed/val_no_morph/*/*.png'
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

for img_name in val_data:
    folders = img_name.split('/')
    label = folders[9]
    label_idx = name2idx[label]
    _, img = boundingboxcrop(img_name)
    #print(img)
    if img != []:
        img = bin_transform(img).numpy()
        pred_label, prob = get_label_probability(img[0], model)

        if pred_label == label_idx:
            cor += 1
        else:
            wro += 1
print(cor/(cor+wro))

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
