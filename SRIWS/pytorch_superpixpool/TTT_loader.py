
import torch.utils.data as data
import albumentations as A
import torch
import pandas as pd
import numpy as np
from PIL import Image
from os.path import join
import os
import imageio
import tifffile as tif
from IRN.imutils import pil_rescale
from IRN.indexing import PathIndex

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
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
]
)
# for test
image_transform_test = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
]
)
# # used for high resolution images 1 m
# IMG_MEAN_ALL = np.array([461.7179, 350.8342, 295.0500, 258.8854])
# IMG_STD_ALL = np.array([135.3398, 116.8479, 117.2889, 108.3638])
# # for uint8
# IMG_MEAN_ALL_low = np.array([64.7314  ,70.5687 ,  74.6582 ,  90.3355])
# IMG_STD_ALL_low = np.array([33.4498 ,  35.7775 ,  38.0263 ,  36.7164])
# for google img, RGB
IMG_MEAN_ALL_Ge = np.array([98.3519, 96.9567, 95.5713])
IMG_STD_ALL_Ge = np.array([52.7343, 45.8798, 44.3465])

# used for high resolution images 1 m
# IMG_MEAN_ALL_Gf = np.array([461.7179, 350.8342, 295.0500, 258.8854])
# IMG_STD_ALL_Gf = np.array([135.3398, 116.8479, 117.2889, 108.3638])
IMG_MEAN_ALL_Gf = np.array([495.742416, 379.526141, 322.051960, 276.685233])
IMG_STD_ALL_Gf = np.array([129.443616, 116.345968, 119.554353, 100.763886])

def norm_totensor(img, mean=IMG_MEAN_ALL_Ge, std=IMG_STD_ALL_Ge, channels=3):
    img = (img - mean[:channels]) / std[:channels]
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
    return img

class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)

class Segmentation_update(data.Dataset):
    def __init__(self, data_dir,superpix_root, channels=3, aug=False):
        self.images_path_list = []
        for root, dirs, files in os.walk(data_dir, topdown=False):
            # for root_remove,dirs_remove,files_remove in os.walk('./runs/jianshe_irnet/gpc_update',topdown = False):
            #     files_final = [i for i in files if i not in files_remove]
            for name in files:
                if (name != "Thumbs.db"):
                    if '.png' in name:
                        self.images_path_list.append(os.path.join(data_dir, name))
        self.aug = aug  # augmentation for images
        self.channels = channels
        self.superpixroot = superpix_root
    def __getitem__(self, item):
        img_path = self.images_path_list[item]
        img = Image.open(img_path).convert('RGB')  # avoid RGBA
        img = np.array(img)  # convert to RGB
        ibase = os.path.basename(img_path)[:-4]
        SuperPix_path = join(self.superpixroot, ibase + '.png')
        SuperPix = np.array(imageio.imread(SuperPix_path))

        if self.aug:
            transformed = image_transform(image=img, mask=SuperPix)
            img = transformed["image"]
            SuperPix = transformed["mask"]
        else:
            transformed = image_transform_test(image=img,mask=SuperPix)
            img = transformed["image"]  # centercrop
            SuperPix = transformed["mask"]
        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        SuperPix = torch.from_numpy(SuperPix)
        return img_path, img, SuperPix
    def __len__(self):
        return len(self.images_path_list)

class myImageFloder_IRN_update(data.Dataset):
    def __init__(self, segroot, datalist,superpix_root, update_path, channels=3, aug=False):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        self.aug = aug  # augmentation for images
        self.channels = channels
        self.segroot = segroot  # for segmentation
        self.update_path = update_path
        self.superpixroot = superpix_root
    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        ibase = os.path.basename(img_path)[:-4]
        img = Image.open(img_path).convert('RGB')  # avoid RGBA
        img = np.array(img)  # convert to RGB
        masks = []

        mask = Image.open(os.path.join(self.segroot, ibase + '.png'))
        mask = np.array(mask)
        update = Image.open(os.path.join(self.update_path, ibase + '.png'))
        update = np.array(update)  # updated img
        # mask = A.center_crop(mask, crop_height=update.shape[0], crop_width=update.shape[1])
        masks.append(mask)
        masks.append(update)
        SuperPix_path = join(self.superpixroot, ibase + '.png')
        SuperPix = np.array(imageio.imread(SuperPix_path))
        masks.append(SuperPix)

        # ref = np.stack((mask, update), axis=2)  # H W C

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, masks=masks)
            img = transformed["image"]
            mask = transformed["masks"][0]
            update = transformed["masks"][1]
            SuperPix = transformed["masks"][2]
        else:
            transformed = image_transform_test(image=img, masks=masks)
            img = transformed["image"]
            mask = transformed["masks"][0]
            update = transformed["masks"][1]
            SuperPix = transformed["masks"][2]

        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        mask = torch.from_numpy(mask)
        update = torch.from_numpy(update)
        SuperPix = torch.from_numpy(SuperPix)

        # mask = torch.from_numpy(ref[:, :, 0]).long()
        # update = torch.from_numpy(ref[:, :, 1]).long()
        return img, mask, update, SuperPix
    def __len__(self):
        return len(self.datalist)
# 2021.11.08 add pseudo labels
class myImageFloder_IRN_pseudo(data.Dataset):
    def __init__(self,  labelroot, datalist, superpix_root, channels=3, aug=False, num_sample = 0, classes = 3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background
        # add
        self.labelroot = labelroot
        self.superpixroot = superpix_root

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB
        masks = []
        ibase = os.path.basename(img_path)[:-4]
        mask_path = join(self.labelroot, ibase+'.png') # mask
        mask = np.array(Image.open(mask_path)).astype('uint8') # 0,1
        masks.append(mask)
        SuperPix_path = join(self.superpixroot, ibase + '.png')
        SuperPix = np.array(imageio.imread(SuperPix_path))
        masks.append(SuperPix)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, masks=masks)
            img = transformed["image"]
            mask = transformed["masks"][0]
            SuperPix = transformed["masks"][1]
        else:
            transformed = image_transform_test(image=img, masks=masks)
            img = transformed["image"] # centercrop
            mask = transformed["masks"][0]
            SuperPix = transformed["masks"][1]
        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask)
        SuperPix = torch.from_numpy(SuperPix)
        return img, mask, SuperPix

    def __len__(self):
        return len(self.datalist)

class Segmentation_update_Gf(data.Dataset):
    def __init__(self, data_dir,superpix_root, channels=4, aug=False):
        self.images_path_list = []
        for root, dirs, files in os.walk(data_dir, topdown=False):
            # for root_remove,dirs_remove,files_remove in os.walk('./runs/jianshe_irnet/gpc_update',topdown = False):
            #     files_final = [i for i in files if i not in files_remove]
            for name in files:
                if (name != "Thumbs.db"):
                    if '.tif' in name:
                        self.images_path_list.append(os.path.join(data_dir, name))
        self.aug = aug  # augmentation for images
        self.channels = channels
        self.superpixroot = superpix_root
    def __getitem__(self, item):
        img_path = self.images_path_list[item]
        img = tif.imread(img_path)
        ibase = os.path.basename(img_path)[:-4]
        SuperPix_path = join(self.superpixroot, ibase + '.png')
        SuperPix = np.array(imageio.imread(SuperPix_path))

        if self.aug:
            transformed = image_transform(image=img, mask=SuperPix)
            img = transformed["image"]
            SuperPix = transformed["mask"]
        else:
            transformed = image_transform_test(image=img,mask=SuperPix)
            img = transformed["image"]  # centercrop
            SuperPix = transformed["mask"]
        img = (img - IMG_MEAN_ALL_Gf[:self.channels]) / IMG_STD_ALL_Gf[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        SuperPix = torch.from_numpy(SuperPix)
        return img_path, img, SuperPix
    def __len__(self):
        return len(self.images_path_list)

class myImageFloder_IRN_update_Gf(data.Dataset):
    def __init__(self, segroot, datalist,superpix_root, update_path, channels=3, aug=False):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        self.aug = aug  # augmentation for images
        self.channels = channels
        self.segroot = segroot  # for segmentation
        self.update_path = update_path
        self.superpixroot = superpix_root
    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        ibase = os.path.basename(img_path)[:-4]
        img = tif.imread(img_path)
        masks = []

        mask = Image.open(os.path.join(self.segroot, ibase + '.png'))
        mask = np.array(mask)
        update = Image.open(os.path.join(self.update_path, ibase + '.png'))
        update = np.array(update)  # updated img
        # mask = A.center_crop(mask, crop_height=update.shape[0], crop_width=update.shape[1])
        masks.append(mask)
        masks.append(update)
        SuperPix_path = join(self.superpixroot, ibase + '.png')
        SuperPix = np.array(imageio.imread(SuperPix_path))
        masks.append(SuperPix)

        # ref = np.stack((mask, update), axis=2)  # H W C

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, masks=masks)
            img = transformed["image"]
            mask = transformed["masks"][0]
            update = transformed["masks"][1]
            SuperPix = transformed["masks"][2]
        else:
            transformed = image_transform_test(image=img, masks=masks)
            img = transformed["image"]
            mask = transformed["masks"][0]
            update = transformed["masks"][1]
            SuperPix = transformed["masks"][2]

        img = (img - IMG_MEAN_ALL_Gf[:self.channels]) / IMG_STD_ALL_Gf[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        mask = torch.from_numpy(mask)
        update = torch.from_numpy(update)
        SuperPix = torch.from_numpy(SuperPix)

        # mask = torch.from_numpy(ref[:, :, 0]).long()
        # update = torch.from_numpy(ref[:, :, 1]).long()
        return img, mask, update, SuperPix
    def __len__(self):
        return len(self.datalist)
# 2021.11.08 add pseudo labels
class myImageFloder_IRN_pseudo_Gf(data.Dataset):
    def __init__(self,  labelroot, datalist, superpix_root, channels=3, aug=False, num_sample = 0, classes = 3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.classes = classes # lvwang, jianshe, background
        # add
        self.labelroot = labelroot
        self.superpixroot = superpix_root

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)
        masks = []
        ibase = os.path.basename(img_path)[:-4]
        mask_path = join(self.labelroot, ibase+'.png') # mask
        mask = np.array(Image.open(mask_path)).astype('uint8') # 0,1
        masks.append(mask)
        SuperPix_path = join(self.superpixroot, ibase + '.png')
        SuperPix = np.array(imageio.imread(SuperPix_path))
        masks.append(SuperPix)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, masks=masks)
            img = transformed["image"]
            mask = transformed["masks"][0]
            SuperPix = transformed["masks"][1]
        else:
            transformed = image_transform_test(image=img, masks=masks)
            img = transformed["image"] # centercrop
            mask = transformed["masks"][0]
            SuperPix = transformed["masks"][1]
        img = (img - IMG_MEAN_ALL_Gf[:self.channels]) / IMG_STD_ALL_Gf[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask)
        SuperPix = torch.from_numpy(SuperPix)
        return img, mask, SuperPix

    def __len__(self):
        return len(self.datalist)

# predict return img path
class myImageFloder_path(data.Dataset):
    def __init__(self,  datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        #self.camroot = camroot

    def __getitem__(self, index):
        img_path = os.path.join(self.datalist.iloc[index, 0])
        img = Image.open(img_path).convert('RGB') # avoid RGBA
        img = np.array(img) # convert to RGB
        mask_path = os.path.join(self.datalist.iloc[index, 1])
        mask = np.array(Image.open(mask_path))
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = image_transform_test(image=img, mask=mask)
            img = transformed["image"] # centercrop
            mask = transformed["mask"]
        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        # mask
        mask = torch.tensor(mask).long()
        return img, mask, img_path

    def __len__(self):
        return len(self.datalist)

class Segmentation_test_Ge(data.Dataset):
    def __init__(self, data_dir, mask_dir,superpix_root, channels=3, aug=False):
        self.images_path_list = []
        self.masks_path_list = []
        self.SuperPix_path_list = []
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for name in files:
                if (name != "Thumbs.db"):
                    self.images_path_list.append(os.path.join(data_dir, name))
                    self.masks_path_list.append(os.path.join(mask_dir,name[:-4]+'.png'))
                    self.SuperPix_path_list.append(os.path.join(superpix_root, name[:-4]+'.png'))
        self.aug = aug  # augmentation for images
        self.channels = channels

    def __getitem__(self, item):
        img_path = self.images_path_list[item]
        img = Image.open(img_path).convert('RGB')  # avoid RGBA
        img = np.array(img)  # convert to RGB
        masks = []
        mask_path = self.masks_path_list[item]
        mask = np.array(Image.open(mask_path))
        masks.append(mask)
        SuperPix_path = self.SuperPix_path_list[item]
        SuperPix = np.array(imageio.imread(SuperPix_path))
        masks.append(SuperPix)
        if self.aug:
            transformed = image_transform(image=img,masks=masks)
            img = transformed["image"]
            mask = transformed["masks"][0]
            SuperPix = transformed["masks"][1]
        else:
            transformed = image_transform_test(image=img,masks=masks)
            img = transformed["image"]  # centercrop
            mask = transformed["masks"][0]
            SuperPix = transformed["masks"][1]
        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        mask = torch.tensor(mask).long()
        SuperPix = torch.from_numpy(SuperPix)
        return img_path, img,mask,SuperPix
    def __len__(self):
        return len(self.images_path_list)

class Segmentation_test_Gf(data.Dataset):
    def __init__(self, data_dir, mask_dir,superpix_root, channels=3, aug=False):
        self.images_path_list = []
        self.masks_path_list = []
        self.SuperPix_path_list = []
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for name in files:
                if (name != "Thumbs.db"):
                    if 'tif' in name:
                        self.images_path_list.append(os.path.join(data_dir, name))
                        self.masks_path_list.append(os.path.join(mask_dir,name[:-4]+'.tif'))
                        self.SuperPix_path_list.append(os.path.join(superpix_root, name[:-4]+'.png'))
        self.aug = aug  # augmentation for images
        self.channels = channels

    def __getitem__(self, item):
        img_path = self.images_path_list[item]
        img = tif.imread(img_path)
        masks = []
        mask_path = self.masks_path_list[item]
        mask = np.array(Image.open(mask_path))
        masks.append(mask)
        SuperPix_path = self.SuperPix_path_list[item]
        SuperPix = np.array(imageio.imread(SuperPix_path))
        masks.append(SuperPix)
        if self.aug:
            transformed = image_transform(image=img,masks=masks)
            img = transformed["image"]
            mask = transformed["masks"][0]
            SuperPix = transformed["masks"][1]
        else:
            transformed = image_transform_test(image=img,masks=masks)
            img = transformed["image"]  # centercrop
            mask = transformed["masks"][0]
            SuperPix = transformed["masks"][1]
        img = (img - IMG_MEAN_ALL_Gf[:self.channels]) / IMG_STD_ALL_Gf[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        mask = torch.tensor(mask).long()
        SuperPix = torch.from_numpy(SuperPix)
        return img_path, img,mask,SuperPix
    def __len__(self):
        return len(self.images_path_list)

class Segmentation_test_Ge_only(data.Dataset):
    def __init__(self, data_dir,superpix_root, channels=3, aug=False):
        self.images_path_list = []
        self.SuperPix_path_list = []
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for name in files:
                if (name != "Thumbs.db"):
                    self.images_path_list.append(os.path.join(data_dir, name))
                    self.SuperPix_path_list.append(os.path.join(superpix_root, name[:-4]+'.png'))
        self.aug = aug  # augmentation for images
        self.channels = channels

    def __getitem__(self, item):
        img_path = self.images_path_list[item]
        img = Image.open(img_path).convert('RGB')  # avoid RGBA
        img = np.array(img)  # convert to RGB

        SuperPix_path = self.SuperPix_path_list[item]
        SuperPix = np.array(imageio.imread(SuperPix_path))
        if self.aug:
            transformed = image_transform(image=img,mask=SuperPix)
            img = transformed["image"]
            SuperPix = transformed["mask"]
        else:
            transformed = image_transform_test(image=img,mask=SuperPix)
            img = transformed["image"]  # centercrop
            SuperPix = transformed["mask"]
        img = (img - IMG_MEAN_ALL_Ge[:self.channels]) / IMG_STD_ALL_Ge[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        SuperPix = torch.from_numpy(SuperPix)
        return img_path, img,SuperPix
    def __len__(self):
        return len(self.images_path_list)

class Segmentation_test_Gf_only(data.Dataset):
    def __init__(self, data_dir,superpix_root, channels=3, aug=False):
        self.images_path_list = []
        self.SuperPix_path_list = []
        for root, dirs, files in os.walk(data_dir, topdown=False):
            for name in files:
                if (name != "Thumbs.db"):
                    if 'tif' in name:
                        self.images_path_list.append(os.path.join(data_dir, name))
                        self.SuperPix_path_list.append(os.path.join(superpix_root, name[:-4]+'.png'))
        self.aug = aug  # augmentation for images
        self.channels = channels

    def __getitem__(self, item):
        img_path = self.images_path_list[item]
        img = tif.imread(img_path)
        SuperPix_path = self.SuperPix_path_list[item]
        SuperPix = np.array(imageio.imread(SuperPix_path))
        if self.aug:
            transformed = image_transform(image=img,mask=SuperPix)
            img = transformed["image"]
            SuperPix = transformed["mask"]
        else:
            transformed = image_transform_test(image=img,mask=SuperPix)
            img = transformed["image"]  # centercrop
            SuperPix = transformed["mask"]
        img = (img - IMG_MEAN_ALL_Gf[:self.channels]) / IMG_STD_ALL_Gf[:self.channels]
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
        SuperPix = torch.from_numpy(SuperPix)
        return img_path, img,SuperPix
    def __len__(self):
        return len(self.images_path_list)

if __name__ =="__main__":
    a = np.ones((256,256), dtype=np.float)
    b = A.rotate(a, 15,)
    print(b.min())