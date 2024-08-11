import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils import data
from Unet_superpixpool import Unet_Superpix
from TTT_loader import Segmentation_test_Ge,Segmentation_test_Gf,Segmentation_test_Ge_only,Segmentation_test_Gf_only
from metrics import SegmentationMetric, acc2file, accprint
import cv2

def main():
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # Setup datalist
    data_dir = r'./datasets_gf/valid/img'
    mask_dir = r'./datasets_gf/valid/lab'
    # Setup parameters
    batch_size = 1
    classes = 2
    nchannels = 4
    device = 'cuda'
    logdir = r'D:\wwr\superpixPool-master\pytorch_superpixpool\runs\regnet040_irnet\pos_pseudov2_Classify_densent169_gf_3'
    SuperPix_dir = r'./datasets_gf/seg_Mynet'
    # test
    testdataloader = torch.utils.data.DataLoader(
        Segmentation_test_Gf(data_dir,mask_dir,SuperPix_dir, channels=nchannels, aug=False),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    net = Unet_Superpix(encoder_name="timm-regnety_040", encoder_weights="imagenet",
                   in_channels=nchannels, classes=2).to(device)

    # print the model
    start_epoch = 0
    best_acc = 0
    for i in range(0,25):
        resume = os.path.join(logdir, 'checkpoint'+str(i+1)+'.tar')
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            net.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
            start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at resume")
            print("=> Will stop.")
            continue

        id = str(start_epoch)
        txtpath = os.path.join(logdir, 'acc' + id + '.txt')  # save acc

        # should be placed after weight loading
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

        # get all parameters (model parameters + task dependent log variances)
        # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
        # weights = torch.FloatTensor([0.6, 0.3]).to(device) # defined
        iou, acc_total = vtest_epoch(net, testdataloader, device, classes, start_epoch, txtpath)
        if best_acc < iou:
            best_acc = iou
            print(best_acc)
            acc2file(acc_total, txtpath)
        # output_dir_seg = os.path.join(r'D:\wwr\Pred\Ablation','Pred_CE')
        # Pred(net,testdataloader,device, output_dir_seg)


def vtest_epoch(model, dataloader, device, classes, epoch, txtpath):
    model.eval()
    # output_dir_seg = r'./datasets_gf/jianshe/pred_seg'
    # output_dir_super = r'./datasets_gf/jianshe/pred_super'
    # os.makedirs(output_dir_seg,exist_ok=True)
    # os.makedirs(output_dir_super, exist_ok=True)
    acc_total = SegmentationMetric(numClass=classes, device=device)
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (img_path, x, y_true,Superpix) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True) # n c h w
            Superpix = Superpix.to(device, non_blocking=True)
            output_seg,output_super= model(x,Superpix)
            # output = torch.softmax(output_seg, dim=1) + torch.softmax(output_super, dim=1)
            # ypred_seg = (torch.sigmoid(output_seg) > 0.5)
            ypred_seg = output_seg.argmax(1)

            acc_total.addBatch(ypred_seg, y_true)
            # ypred_seg= ypred_seg.squeeze().cpu().numpy().astype(np.int16)

            # cv2.imwrite(os.path.join(output_dir_seg, img_path[0].split('\\')[-1]), ypred_seg)
            # ypred_seg[ypred_seg == 1] = 255
            # cv2.imwrite(os.path.join(output_dir_seg, img_path[0].split('\\')[-1][:-4] + '_c.png'), ypred_seg)

            # ypred_super = ypred_super.squeeze().cpu().numpy().astype(np.int16)

            # cv2.imwrite(os.path.join(output_dir_super, img_path[0].split('\\')[-1]), ypred_super)
            # ypred_super[ypred_super == 1] = 255
            # cv2.imwrite(os.path.join(output_dir_super, img_path[0].split('\\')[-1] + '_c.png'), ypred_super)
            f1 = acc_total.F1score()[1]
            iou = acc_total.IntersectionOverUnion()[1]
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. F1 {f1:.3f}, IOU: {iou:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, f1=f1, iou=iou))
            pbar.update()
        pbar.close()
    #
    accprint(acc_total)
    return iou,acc_total

def Pred(model, dataloader, device, output_dir_seg):
    model.eval()
    os.makedirs(output_dir_seg,exist_ok=True)
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (img_path, x, Superpix) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            # y_true = y_true.to(device, non_blocking=True)  # n c h w
            Superpix = Superpix.to(device, non_blocking=True)
            output_seg, output_super = model(x, Superpix)
            ypred_seg = output_seg.argmax(1)
            cv2.imwrite(os.path.join(output_dir_seg, img_path[0].split('\\')[-1]), ypred_seg[0].cpu().numpy())
            ypred_seg[ypred_seg == 1] = 255
            cv2.imwrite(os.path.join(output_dir_seg, img_path[0].split('\\')[-1][:-4] + '_c.png'), ypred_seg[0].cpu().numpy())
if __name__ == "__main__":
    main()