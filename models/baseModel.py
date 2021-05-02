import os
import PIL
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dataset = None

    def getData(self, dir):
        self.dataset = PhotoDataset(dir)
    
    def load_checkpoint(self, ckpt_path, map_location=None):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
        return ckpt

    def save_checkpoint(self, state, save_path):
        torch.save(state, save_path)


class PhotoDataset(Dataset):
    def __init__(self, photo_dir):
        super().__init__()
        self.photo_dir = photo_dir
        self.photo_idx = dict()
        
        for i, fl in enumerate(sorted(os.listdir(self.photo_dir))):
            self.photo_idx[i] = fl
        self.idx = 1
    def __getitem__(self, idx):
        photo_path = os.path.join(self.photo_dir, self.photo_idx[idx])
        photo_img = Image.open(photo_path)
        return photo_img

    def __len__(self):
        return len(self.photo_idx.keys())