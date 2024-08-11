import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import torch
torch.set_num_threads(2)
import random
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from Unet_superpixpool import Unet_Superpix
from torch.utils import data
import torch.nn.functional as F
from tensorboardX import SummaryWriter #change tensorboardX
from TTT_loader import myImageFloder_IRN_pseudo_Gf, Segmentation_update_Gf,myImageFloder_IRN_update_Gf
from metrics import SegmentationMetric, AverageMeter

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 5:
        lr = 0.001
    elif epoch <= 15:
        lr = 0.0001
    else:
        lr = 0.001
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr #added
def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)
def loss_jocor(y1, y2, labels, forget_rate, ignore_index, co_lambda):
    loss_pick_1 = F.cross_entropy(y1, labels, reduce=False, ignore_index=ignore_index) * (1 - co_lambda)
    loss_pick_2 = F.cross_entropy(y2, labels, reduce=False, ignore_index=ignore_index) * (1 - co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y1, y2,reduce=False) + co_lambda * kl_loss_compute(y2, y1, reduce=False))
    loss_pick[labels == ignore_index] = 0
    loss = torch.mean(loss_pick)

    return loss

def main():
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # Setup datalist
    trainlist_pos = r"./datasets_gf/train_list_0.6_jianshe_pos.txt"
    segroot_warm = r'./runs/jianshe_irnet_densent169_gf'
    segroot_update = os.path.join(segroot_warm, "gpc_update")
    os.makedirs(segroot_update, exist_ok=True)
    segroot_final = os.path.join(segroot_warm,"gpc_final")
    os.makedirs(segroot_final,exist_ok=True)
    testlablist = r'./datasets_gf/test_list_0.6_jianshe.txt'
    SuperPix_dir = r'./datasets_gf/seg_Mynet'
    # Setup paramters
    imgsize = 512
    batch_size = 6
    epochs = 25

    classes = 2
    nchannels = 4
    device = 'cuda'
    logdir = r'.\runs\regnet040_irnet\pos_pseudov2_Classify_densent169_gf_3'
    global best_acc
    best_acc = 0
    warm_up = 6
    update = 10

    final = 10
    writer = SummaryWriter(log_dir=logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # train & test dataloader
    traindataloader_warm = torch.utils.data.DataLoader(
        myImageFloder_IRN_pseudo_Gf(segroot_warm, trainlist_pos, SuperPix_dir, aug=True, channels=nchannels),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_update = torch.utils.data.DataLoader(
        Segmentation_update_Gf(data_dir=r'D:\wwr\lvwang\gf2\jianshe_final', superpix_root=SuperPix_dir),
        batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True
    )
    traindataloader_update = torch.utils.data.DataLoader(
        myImageFloder_IRN_update_Gf(segroot_warm,trainlist_pos, SuperPix_dir,segroot_update,aug=True,channels=nchannels),
        batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True
    )
    traindataloader_final = torch.utils.data.DataLoader(
        myImageFloder_IRN_pseudo_Gf(segroot_final, trainlist_pos, SuperPix_dir,aug=True, channels=nchannels),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    testdataloader = torch.utils.data.DataLoader(
        myImageFloder_IRN_pseudo_Gf(segroot_warm, testlablist, SuperPix_dir,aug=False, channels=nchannels),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    net = Unet_Superpix(encoder_name="timm-regnety_040", encoder_weights="imagenet",
                   in_channels=nchannels, classes=2).to(device)


    # should be placed after weight loading
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    start_epoch = 0
    resume = os.path.join(logdir, 'checkpoint20.tar')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        # best_acc = checkpoint['best_acc']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will stop.")

    for epoch in range(epochs - start_epoch):
        epoch = start_epoch + epoch + 1  # current epochs
        adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        print('epoch %d, lr: %.6f' % (epoch, lr))
        if epoch < warm_up:
            train_loss, train_oa, train_iou = train_epoch_warm(net,
                                                          traindataloader_warm,
                                                          optimizer, device, epoch, classes)
        elif epoch < (warm_up + update):
            update_label(net,dataloader_update, device, segroot_update)
            train_loss, train_oa, train_iou = train_epoch_update(net,traindataloader_update,optimizer,
                                                                 device,epoch,classes)
        elif epoch < (warm_up + update + final):
            if epoch < (warm_up + update + 1):
                final_label(net,dataloader_update,device,segroot_final)
                train_loss, train_oa, train_iou = train_epoch_scratch(net,
                                                          traindataloader_final,
                                                          optimizer, device, epoch, classes)
            else:
                train_loss, train_oa, train_iou = train_epoch_scratch(net,
                                                                      traindataloader_final,
                                                                      optimizer, device, epoch, classes)
            # val_loss, val_oa, val_iou = vtest_epoch(net, criterion, testdataloader, device, epoch, classes)
        # save every epoch
        savefilename = os.path.join(logdir, 'checkpoint' + str(epoch) + '.tar')
        # is_best = val_iou[1] > best_acc
        # best_acc = max(val_iou[1], best_acc)  # update
        torch.save({
            'epoch': epoch,
            'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
            # 'val_oa': val_oa,
            'optimizer' : optimizer.state_dict(),
        }, savefilename)
        # if is_best:
        #     shutil.copy(savefilename, os.path.join(logdir, 'model_best.tar'))
        # write
        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('train/1.loss', train_loss, epoch)
        writer.add_scalar('train/2.oa', train_oa, epoch)
        writer.add_scalar('train/3.iou_lvwang', train_iou[1], epoch)
        writer.add_scalar('train/4.iou_fei', train_iou[0], epoch)
        # writer.add_scalar('val/1.loss', val_loss, epoch)
        # writer.add_scalar('val/2.oa', val_oa, epoch)
        # writer.add_scalar('val/3.iou_lvwang', val_iou[1], epoch)
        # writer.add_scalar('val/4.iou_fei', val_iou[0], epoch)
    writer.close()


# train updated labels from scratch
def train_epoch_warm(net, dataloader, optimizer, device, epoch, classes):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (images, update, Superpix) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        update = update.to(device, non_blocking=True)  # N 1 H W
        Superpix = Superpix.to(device, non_blocking=True)
        output_seg,output_super = net(images, Superpix)


        loss = loss_jocor(output_seg,output_super,update.long(),forget_rate=0, ignore_index=255, co_lambda=0.05)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_seg.argmax(1)  # N C H W
        acc_total.addBatch(output[update != 255], update[update != 255])
        losses.update(loss.item(), images.size(0))

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, IOU: {lv:.3f}, {fei:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, lv=iou[1], fei=iou[0]))
        pbar.update()
    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou

def train_epoch_update(net, dataloader, optimizer, device, epoch, classes, alpha=0.5):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (images, mask, update, Superpix) in enumerate(dataloader):
        images = images.to(device, non_blocking=True) # N C H W
        mask = mask.to(device, non_blocking=True)# N 1 H W
        update = update.to(device, non_blocking=True)
        Superpix = Superpix.to(device, non_blocking=True)
        output_seg,output_super = net(images,Superpix)

        loss = loss_jocor(output_seg,output_super,mask.long(),forget_rate=0, ignore_index=255, co_lambda=0.05) \
               + loss_jocor(output_seg,output_super,update.long(),forget_rate=0, ignore_index=255, co_lambda=0.05)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_seg.argmax(1) # N C H W
        acc_total.addBatch(output[update!=255], update[update!=255])
        losses.update(loss.item(), images.size(0))

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, IOU: {gpc:.3f}, {nongpc:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, gpc=iou[1], nongpc=iou[0]))
        pbar.update()
    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou

def train_epoch_scratch(net, dataloader, optimizer, device, epoch, classes):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (images, update, Superpix) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        update = update.to(device, non_blocking=True)  # N 1 H W
        Superpix = Superpix.to(device, non_blocking=True)
        output_seg,output_super = net(images,Superpix)
        loss = loss_jocor(output_seg,output_super,update.long(),forget_rate=0, ignore_index=255, co_lambda=0.2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # output_seg_pred = (torch.sigmoid(output_seg) > 0.5)
        output_seg_pred = output_seg.argmax(1).long()
        # output_super_pred = output_super.argmax(1).long()  # N C H W
        acc_total.addBatch(output_seg_pred[update != 255], update[update != 255])
        losses.update(loss.item(), images.size(0))

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, IOU: {lv:.3f}, {fei:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, lv=iou[1], fei=iou[0]))
        pbar.update()
    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou

def vtest_epoch(model, criterion, dataloader, device, epoch, classes):
    model.eval()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True) # n c h w
            ypred = model.forward(x)

            loss = criterion(ypred, y_true.long())

            # ypred = ypred.argmax(axis=1)
            ypred = ypred.argmax(1)
            acc_total.addBatch(ypred[y_true != 255], y_true[y_true != 255])

            losses.update(loss.item(), x.size(0))
            oa = acc_total.OverallAccuracy()
            iou = acc_total.IntersectionOverUnion()
            miou = acc_total.meanIntersectionOverUnion()
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU{miou:.3f}, IOU: {lv:.3f}, {fei:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, lv=iou[1], fei=iou[0]))
            pbar.update()
        pbar.close()

    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou

def update_label(net, dataloader, device, update_path):
    net.eval()
    with torch.no_grad():
        for imgpath,images,Superpix in tqdm(dataloader):
            images = images.to(device, non_blocking=True) # N C H W
            Superpix = Superpix.to(device, non_blocking=True)
            # mask = mask # N H W
            output_seg,output_super = net(images,Superpix)

            output_seg_pred = output_seg.argmax(1).long()
            output_super_pred = output_super.argmax(1).long()
            output_pred = torch.ones(output_seg_pred.shape) * 255
            output_pred[(output_super_pred == 1) & (output_seg_pred ==1)] = 1
            output_pred[(output_super_pred == 0) & (output_seg_pred == 0)] = 0
            # save
            for idx, imgp in enumerate(imgpath):
                ibase = os.path.basename(imgp)[:-4]
                resname = os.path.join(update_path, ibase)
                tmp = output_pred[idx].squeeze().cpu().numpy().astype('uint8')# H W, [0,1]

                cv2.imwrite(resname+'.png', tmp)
                cv2.imwrite(resname+'_c.png', tmp*255)

def final_label(net, dataloader, device, final_path):
    net.eval()
    with torch.no_grad():
        for imgpath,images,Superpix in tqdm(dataloader):
            images = images.to(device, non_blocking=True) # N C H W
            Superpix = Superpix.to(device, non_blocking=True)
            # mask = mask # N H W
            output_seg,output_super = net(images,Superpix)

            output_seg_pred = output_seg.argmax(1).long()
            # save
            for idx, imgp in enumerate(imgpath):
                ibase = os.path.basename(imgp)[:-4]
                resname = os.path.join(final_path, ibase)
                tmp = output_seg_pred[idx].squeeze().cpu().numpy().astype('uint8')# H W, [0,1]
                cv2.imwrite(resname+'.png', tmp)
                cv2.imwrite(resname+'_c.png', tmp*255)

if __name__ == '__main__':
    main()