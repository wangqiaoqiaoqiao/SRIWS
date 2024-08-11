## generate cues from grad-cam++
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from tqdm import tqdm
import torch.nn.functional as F
from gradcam.grad_cam_plusplus import GradCAMPlusPlus
import numpy as np
from TTT_loader_Classify import Classify_path_Ge,Classify_path_Gf

import tifffile as tif
from PIL import Image

IMG_MEAN = np.array([98.3519, 96.9567, 95.5713])
IMG_STD = np.array([52.7343, 45.8798, 44.3465])
import torch
import torch.nn as nn
from encoders import get_encoder
from base import initialization as init
from typing import Optional
from base import ClassificationHead


class ClassificationModel(torch.nn.Module):
    def __init__(self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            pooling: str = "avg",
            dropout: float = 0.2,):
        super(ClassificationModel, self).__init__()
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1], classes=classes, pooling=pooling, dropout=dropout, activation=activation)
        self.initialize() # initialize head

    def initialize(self):
        init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        labels = self.classification_head(features[-1])
        return labels



def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img = (img-IMG_MEAN)/IMG_STD
    img = torch.from_numpy(img).permute(2, 0, 1).float()  # H W C ==> C H W
    return img.unsqueeze(0)


# generate grad-cam
def gen_cam_jianshe(model, dataloader, train_cues_dir, backbone='regnet',
         device='cuda', aug_smooth = False, eigen_smooth=False):
    os.makedirs(train_cues_dir, exist_ok=True)
    # os.makedirs(os.path.join(train_cues_dir, 'jianshe_cammask'), exist_ok=True)
    localization_cues = {}#
    target_layer=''
    if backbone=="regnet":
        target_layer = [model.encoder.s4]
    elif backbone=="resnet":
        target_layer = [model.encoder.layer4[-1]]
    elif backbone == "densenet":
        target_layer = [model.encoder.features.denseblock4]
    elif backbone == "vgg":
        target_layer = [model.encoder.features[-1]]
    else:
        raise Exception("Error in set target_layer")

    cam_method = GradCAMPlusPlus(model=model, target_layers=target_layer,
                    use_cuda=True if device == "cuda" else False)
    # Process by batch
    model.eval()
    for img, imgpath in tqdm(dataloader):
        # img = preprocess(imgpath)
        img = img.to(device, non_blocking=True)
        pred_scores = model(img)
        pred_scores = F.softmax(pred_scores, dim=1) # N C
        pred_scores = pred_scores.cpu().detach().numpy() # N C
        # generate grad-cam for a batch of img. H: shape (N C H W)
        # xsize = img.shape[0]
        # run a batch of imgs and for all classes
        H = cam_method(input_tensor=img, target_category=[0],
                       aug_smooth=aug_smooth, eigen_smooth=eigen_smooth)
        # save grad_cam and pred_scores
        for i, imp in enumerate(imgpath):
            iname = os.path.basename(imp)[:-4]
            idir = os.path.basename(os.path.dirname(os.path.dirname(imp)))# "lvwang
            icue = os.path.join(train_cues_dir, iname)
            j=0
            h = H[i, :, :] # N H W C
            tif.imwrite(icue+'_%d_%.3f.tif'%(j, pred_scores[i,j]), h) #
            # shutil.copy(imp, icue+'.png')


def main():
    #
    train_cues_dir = r'./runs/pred_cam_CAM_Atten_MIX_1_densenet169_gf'

    nchannels = 4
    classes = 2
    device ='cuda'

    train_list_pos = r'./datasets_gf/list_jianshe_pos.txt'
    train_dataset = Classify_path_Gf(train_list_pos, channels=nchannels, aug=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    # use balance model
    net = ClassificationModel(encoder_name="densenet169", encoder_weights="imagenet",
                             in_channels=nchannels, classes=classes).to(device)
    pretrainp = r'./runs/Classify_CAM_Atten_MIX1_densenet169_gf/model_best.tar'
    if not os.path.exists(pretrainp):
        return
    net.load_state_dict(torch.load(pretrainp)["state_dict"])
    net.eval() # keep the batchnorm layer constant

    # target: 0==>gpc
    gen_cam_jianshe(model=net, dataloader=train_loader, train_cues_dir=train_cues_dir,
             backbone='densenet', device=device, aug_smooth=False, eigen_smooth=False)

if __name__=="__main__":
    main()