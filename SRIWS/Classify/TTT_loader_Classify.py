'''
2021.9.6 tuitiantu
'''
import tifffile as tif
import torch.utils.data as data
import albumentations as A
import torch
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os

# imgsize for google
imgsize = 512
image_transform = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
    A.Flip(p=0.5),
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
]
)
# for train
image_transform_randomcrop = A.Compose([
    A.RandomCrop(width=imgsize, height=imgsize, always_apply=True),
    A.Flip(p=0.5),
   # A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
]
)
# for test
image_transform_test = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
]
)
# used for high resolution images 1 m
# IMG_MEAN_ALL_Gf = np.array([461.7179, 350.8342, 295.0500, 258.8854])
# IMG_STD_ALL_Gf = np.array([135.3398, 116.8479, 117.2889, 108.3638])
IMG_MEAN_ALL_Gf = np.array([495.742416, 379.526141, 322.051960, 276.685233])
IMG_STD_ALL_Gf = np.array([129.443616, 116.345968, 119.554353, 100.763886])

# GE Dataset for google img, RGB
IMG_MEAN_ALL_Ge = np.array([98.3519, 96.9567, 95.5713])
IMG_STD_ALL_Ge = np.array([52.7343, 45.8798, 44.3465])


def norm_totensor(img, mean=IMG_MEAN_ALL_Ge, std=IMG_STD_ALL_Ge, channels=3):
    img = (img - mean[:channels]) / std[:channels]
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
    return img
class Classify_path_Ge(data.Dataset):
    def __init__(self, data_path, channels = 3, aug = False):
        # self.images_path_list = []
        self.images_path_list = pd.read_csv(data_path, sep=',', header=None)
        self.aug = aug
        self.channels = channels
        # for root, dirs, files in os.walk(data_path, topdown=False):
        #     for name in files:
        #         if (name != "Thumbs.db"):
        #             if '.png' in name:
        #                 self.images_path_list.append(os.path.join(data_path, name))
    def __getitem__(self, item):
        # img_path = self.images_path_list[item]
        img_path = self.images_path_list.iloc[item, 0]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        if self.aug:
            img = image_transform(image=img)['image']
        else:
            img = image_transform_test(image=img)['image']
        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        return img,img_path

    def __len__(self):
        return len(self.images_path_list)

class Classify_path_Gf(data.Dataset):
    def __init__(self, data_path, channels = 4, aug = False):
        # self.images_path_list = []
        self.images_path_list = pd.read_csv(data_path, sep=',', header=None)
        self.aug = aug
        self.channels = channels
        # for root, dirs, files in os.walk(data_path, topdown=False):
        #     for name in files:
        #         if (name != "Thumbs.db"):
        #             if '.png' in name:
        #                 self.images_path_list.append(os.path.join(data_path, name))
    def __getitem__(self, item):
        # img_path = self.images_path_list[item]
        img_path = self.images_path_list.iloc[item, 0]
        img = tif.imread(img_path)
        if self.aug:
            img = image_transform(image=img)['image']
        else:
            img = image_transform_test(image=img)['image']
        img = (img - IMG_MEAN_ALL_Gf[:self.channels]) / IMG_STD_ALL_Gf[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        return img,img_path

    def __len__(self):
        return len(self.images_path_list)

class myImageFloder_Ge(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.open(img_path).convert('RGB') # avoid RGBA
        # img = np.array(img) # convert to RGB
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]
        else:
            img = image_transform_test(image=img)["image"] # centercrop
        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        return img, lab

    def __len__(self):
        return len(self.datalist)

class myImageFloder_Gf(data.Dataset):
    def __init__(self, datalist, channels=4, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]
        else:
            img = image_transform_test(image=img)["image"] # centercrop
        img = (img - IMG_MEAN_ALL_Gf[:self.channels]) / IMG_STD_ALL_Gf[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        return img, lab

    def __len__(self):
        return len(self.datalist)

