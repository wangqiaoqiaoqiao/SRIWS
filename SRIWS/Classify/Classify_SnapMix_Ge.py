import os

import tifffile

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import torch
import random
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
torch.set_num_threads(4)
from SnapMix import snapmix
from torch.utils import data
from scipy.stats import beta
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter #change tensorboardX
from TTT_loader_Classify import myImageFloder_Ge,myImageFloder_Gf
from model import ClassificationModel
from metrics import ClassificationMetric, AverageMeter
import shutil
IMG_MEAN_ALL = np.array([98.3519, 96.9567, 95.5713])
IMG_STD_ALL = np.array([52.7343, 45.8798, 44.3465])


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 40:
        lr = 0.001
    elif epoch <= 60:
        lr = 0.0001
    else:
        lr = 0.00001
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr #added

def main():
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # Setup datalist
    iroot = r'./datasets_ge'
    trainlist_pos = os.path.join(iroot, 'train_list_0.6_jianshe_pos.txt')
    trainlist_neg = os.path.join(iroot, 'train_list_0.6_jianshe_neg_revise.txt')
    testlist = os.path.join(iroot, 'test_list_0.6_jianshe_revise.txt')
    trainlist_mix_pos = os.path.join(iroot, 'train_list_0.6_jianshe_pos_mix.txt')
    flag = 'CAM_Atten_MIX'
    # Setup parameters
    batch_size = 12
    epochs = 80
    epoch_warm = 2
    classes = 2 #
    nchannels = 3 # channels
    device = 'cuda'
    logdir = r'.\runs\Classify_CAM_Atten_MIX1_densenet169_ge'
    mix_dir = os.path.join(logdir,'mix')
    global best_acc
    best_acc = 0
    writer = SummaryWriter(log_dir=logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(mix_dir):
        os.makedirs(mix_dir)

    # train & test dataloader
    traindataloader_pos = torch.utils.data.DataLoader(
        myImageFloder_Ge(trainlist_pos, aug=True, channels=nchannels),
        batch_size=batch_size//2, shuffle=True, num_workers=4, pin_memory=True)
    traindataloader_neg = torch.utils.data.DataLoader(
        myImageFloder_Ge(trainlist_neg, aug=True, channels=nchannels),
        batch_size=batch_size//2, shuffle=True, num_workers=4, pin_memory=True)

    traindataloader_pos_sample = torch.utils.data.DataLoader(
        myImageFloder_Ge(trainlist_pos, aug=False, channels=nchannels),
        batch_size=batch_size // 2, shuffle=False, num_workers=4, pin_memory=True)
    traindataloader_neg_sample = torch.utils.data.DataLoader(
        myImageFloder_Ge(trainlist_neg, aug=False, channels=nchannels),
        batch_size=batch_size // 2, shuffle=False, num_workers=4, pin_memory=True)

    testdataloader = torch.utils.data.DataLoader(
        myImageFloder_Ge(testlist, aug=False, channels=nchannels),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    net = ClassificationModel(encoder_name="densenet169", encoder_weights="imagenet",
                             in_channels=nchannels, classes=classes).to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001,weight_decay=1e-4,momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    resume = os.path.join(logdir, 'model_besttt.tar')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        # best_acc = checkpoint['best_acc']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")
        # return

    # should be placed after weight loading
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    # get all parameters (model parameters + task dependent log variances)
    # print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    for epoch in range(epochs - start_epoch):
        epoch = start_epoch + epoch + 1  # current epochs
        adjust_learning_rate(optimizer, epoch)
        lr = optimizer.param_groups[0]['lr']
        print('epoch %d, lr: %.6f' % (epoch, lr))
        if flag == 'CAM_Atten_MIX':
            if epoch<epoch_warm:
                print('Warm Up')
                train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos,
                                                             traindataloader_neg,
                                                             optimizer, device, epoch, classes)
            else:
                if epoch< epoch_warm+1:
                    print('Train')
                    CAM_Atten_MIX(net, traindataloader_pos_sample, traindataloader_neg_sample, device, mix_dir, trainlist_mix_pos)
                    shutil.copy(trainlist_pos,trainlist_mix_pos)
                    Note = open(trainlist_mix_pos, mode='a')
                    for root, dirs, files in os.walk(mix_dir, topdown=False):
                        for name in files:
                            if (name != "Thumbs.db"):
                                    image_path = os.path.join(mix_dir,name)
                                    if os.path.exists(image_path):
                                        Note.write('\n'+image_path + ',0')
                    Note.close()
                    traindataloader_mix_pos = torch.utils.data.DataLoader(
                        myImageFloder_Ge(trainlist_mix_pos, aug=True, channels=nchannels),
                        batch_size=batch_size // 2, shuffle=True, num_workers=4, pin_memory=True)
                    train_loss, train_oa, train_f1 = train_epoch(net, criterion,traindataloader_mix_pos,
                                                                 traindataloader_neg,
                                                                 optimizer, device, epoch, classes)
                else:
                    print('Train')
                    CAM_Atten_MIX(net, traindataloader_pos_sample, traindataloader_neg_sample, device, mix_dir, trainlist_mix_pos)
                    train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_mix_pos,
                                                                 traindataloader_neg,
                                                                 optimizer, device, epoch, classes)
        elif flag == 'MIXUP':
            if epoch < epoch_warm:
                print('Warm Up')
                train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos,
                                                             traindataloader_neg,
                                                             optimizer, device, epoch, classes)
            else:
                print('Train')
                print(flag)
                # MIX_UP(net, traindataloader_pos_sample,traindataloader_neg_sample, device, alpha,
                #                                                  mix_dir,trainlist_pos, trainlist_mix_pos)
                # traindataloader_mix_pos = torch.utils.data.DataLoader(
                #     myImageFloder(trainlist_mix_pos, aug=True, channels=nchannels),
                #     batch_size=batch_size // 2, shuffle=True, num_workers=4, pin_memory=True)
                train_loss, train_oa, train_f1 = train_epoch_mix(net, criterion, traindataloader_pos,
                                                             traindataloader_neg,
                                                             optimizer, device, epoch, classes, flag)
                train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos,
                                                             traindataloader_neg,
                                                             optimizer, device, epoch, classes)
        elif flag == 'CUTMIX':
            if epoch < epoch_warm:
                print('Warm Up')
                train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos,
                                                         traindataloader_neg,
                                                         optimizer, device, epoch, classes)
            else:
                print('Train')
                print(flag)
                train_loss, train_oa, train_f1 = train_epoch_mix(net, criterion, traindataloader_pos,
                                                                 traindataloader_neg,
                                                                 optimizer, device, epoch, classes, flag)
                train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos,
                                                             traindataloader_neg,
                                                             optimizer, device, epoch, classes)
        elif flag == 'SNAPMIX':
            if epoch < epoch_warm:
                print('Warm Up')
                train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos,
                                                             traindataloader_neg,
                                                             optimizer, device, epoch, classes)
            else:
                print('Train')
                print(flag)
                train_loss, train_oa, train_f1 = train_epoch_mix(net, criterion, traindataloader_pos,
                                                         traindataloader_neg,
                                                         optimizer, device, epoch, classes, flag)
                train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos,
                                                     traindataloader_neg,
                                                     optimizer, device, epoch, classes)
        elif flag == 'FMIX':
            if epoch < epoch_warm:
                print('Warm Up')
                train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos,
                                                             traindataloader_neg,
                                                             optimizer, device, epoch, classes)
            else:
                print('Train')
                print(flag)
                train_loss, train_oa, train_f1 = train_epoch_mix(net, criterion, traindataloader_pos,
                                                                 traindataloader_neg,
                                                                 optimizer, device, epoch, classes, flag)
                train_loss, train_oa, train_f1 = train_epoch(net, criterion, traindataloader_pos,
                                                             traindataloader_neg,
                                                             optimizer, device, epoch, classes)

        # validate every epoch
        val_loss, val_oa, val_f1= vtest_epoch(net, criterion, testdataloader, device, epoch, classes)
        print('Val OA:{}'.format(val_oa))
        print('Val f1:{}'.format(val_f1))
        # save every epoch
        savefilename = os.path.join(logdir, 'checkpoint' + str(epoch) + '.tar')
        is_best = val_oa > best_acc
        best_acc = max(val_oa, best_acc)  # update
        torch.save({
            'epoch': epoch,
            'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
            'val_oa': val_oa,
            'val_f1': val_f1,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, savefilename)
        if is_best:
            print('best:',best_acc)
            shutil.copy(savefilename, os.path.join(logdir, 'model_best.tar'))
        # write
        writer.add_scalar('lr', lr, epoch)
        # writer.add_scalar('train/1.loss', train_loss, epoch)
        # writer.add_scalar('train/2.oa', train_oa, epoch)
        # writer.add_scalar('train/3.f1_gpc', train_f1[0], epoch)
        # writer.add_scalar('train/4.f1_nongpc', train_f1[1], epoch)
        writer.add_scalar('val/1.loss', val_loss, epoch)
        writer.add_scalar('val/2.oa', val_oa, epoch)
        writer.add_scalar('val/3.f1_gpc', val_f1[0], epoch)
        writer.add_scalar('val/4.f1_nongpc', val_f1[1], epoch)

    writer.close()

def train_epoch(net, criterion, dataloader_pos, dataloader_neg, optimizer, device, epoch, classes):
    net.train()
    acc_total = ClassificationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    # with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
    num = len(dataloader_pos)
    pbar = tqdm(range(num), disable=False)
    neg_train_iter = iter(dataloader_neg) # negative samples
    for idx, (x1, y1) in enumerate(dataloader_pos):
        try:
            x2, y2 = neg_train_iter.next()
        except:
            neg_train_iter = iter(dataloader_neg)
            x2, y2 = neg_train_iter.next()
        # combine pos and negtor
        x = torch.cat((x1, x2), dim=0).to(device, non_blocking=True) # N C H W
        y_true = torch.cat((y1, y2), dim=0).to(device, non_blocking=True) # N H W

        ypred,_ = net(x)

        loss = criterion(ypred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ypred = ypred.argmax(1)
        acc_total.addBatch(ypred, y_true)

        losses.update(loss.item(), x.size(0))
        oa = acc_total.OverallAccuracy()
        f1 = acc_total.F1score()
        pbar.set_description('Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, F1: {lv:.3f}, {fei:.3f}'.format(
                             epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa = oa,lv = f1[0], fei=f1[1]))
        pbar.update()
    pbar.close()
    return losses.avg, acc_total.OverallAccuracy(), acc_total.F1score()

def train_epoch_mix(net, criterion, dataloader_pos, dataloader_neg, optimizer, device, epoch, classes, flag, alpha = 1):
    net.train()
    acc_total1 = ClassificationMetric(numClass=classes, device=device)
    acc_total2 = ClassificationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    # with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
    num = len(dataloader_pos)
    pbar = tqdm(range(num), disable=False)
    neg_train_iter = iter(dataloader_neg) # negative samples
    for idx, (x1, y1) in enumerate(dataloader_pos):
        try:
            x2, y2 = neg_train_iter.next()
        except:
            neg_train_iter = iter(dataloader_neg)
            x2, y2 = neg_train_iter.next()
        # combine pos and negtor
        x = torch.cat((x1, x2), dim=0).to(device, non_blocking=True) # N C H W
        y_true = torch.cat((y1, y2), dim=0).to(device, non_blocking=True) # N H W

        if flag == 'MIXUP':
            inputs, targets_a, targets_b, lam = mixup_data(x, y_true, alpha, True)
            lama = lam
            lamb = 1-lam
        elif flag == 'CUTMIX':
            inputs, targets_a, targets_b, lam = cutmix_data(x, y_true, beta=1, cutmix_prob=1)
            lama = lam
            lamb = 1 - lam
        elif flag == 'SNAPMIX':
            inputs, targets_a, targets_b, lama, lamb = snapmix_data(x, y_true, prob=1.0, beta=1.0,
                                                                    cropsize=512, netname='regnet', model=net)
        elif flag == 'FMIX':
            inputs, targets_a, targets_b, lam = fmix_data(x, y_true)
            lama = lam
            lamb = 1 - lam
            # for i in range(inputs.size(0)):
            #     image = inputs[i].permute(1, 2, 0).detach().cpu().numpy() * IMG_STD_ALL + IMG_MEAN_ALL
            #     imageio.imsave(r'.\runs\Classify_FMIX\mix' + '//' + str(idx) + '_' + str(i) + '_1.png', image.astype('uint8'))
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        ypred, _ = net(inputs)
        loss = mixup_criterion(criterion, ypred, targets_a, targets_b, lama, lamb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ypred = ypred.argmax(1)
        acc_total1.addBatch(ypred, targets_a)
        acc_total2.addBatch(ypred,targets_b)

        losses.update(loss.item(), x.size(0))

        pbar.set_description('Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. '
                             'OA_1 {oa1:.3f}, F1_1: {lv1:.3f}, {fei1:.3f}, '
                             'OA_2 {oa2:.3f}, F1_2: {lv2:.3f}, {fei2:.3f}'.format(
                             epoch=epoch, batch=idx, iter=num, loss=losses.avg,
                    oa1 = acc_total1.OverallAccuracy(),lv1 = acc_total1.F1score()[0], fei1=acc_total1.F1score()[1],
                    oa2 = acc_total2.OverallAccuracy(),lv2 = acc_total2.F1score()[0], fei2=acc_total2.F1score()[1]))
        pbar.update()
    pbar.close()
    return losses.avg, acc_total1.OverallAccuracy(), acc_total1.F1score()

def CAM_Atten_MIX(model, dataloader_pos, dataloader_neg, device, sample_path, sample_txt_path):
    model.eval()
    with torch.no_grad():
        neg_train_iter = iter(dataloader_neg)  # negative samples
        idx = 0
        for x1, y1 in tqdm(dataloader_pos):
            try:
                x2, y2 = neg_train_iter.next()
                # x3, y3 = neg_train_iter.next()
                # x4, y4 = neg_train_iter.next()
                # x5, y5 = neg_train_iter.next()
                if x1.size() != x2.size():
                    neg_train_iter = iter(dataloader_neg)
                    x2, y2 = neg_train_iter.next()
                #     x3, y3 = neg_train_iter.next()
                #     x4, y4 = neg_train_iter.next()
                #     x5, y5 = neg_train_iter.next()
                # elif x1.size() != x3.size():
                #     neg_train_iter = iter(dataloader_neg)
                #     x3, y3 = neg_train_iter.next()
                #     x4, y4 = neg_train_iter.next()
                #     x5, y5 = neg_train_iter.next()
                # elif x1.size() != x4.size():
                #     neg_train_iter = iter(dataloader_neg)
                #     x4, y4 = neg_train_iter.next()
                #     x5, y5 = neg_train_iter.next()
                # elif x1.size() != x5.size():
                #     neg_train_iter = iter(dataloader_neg)
                #     x5, y5 = neg_train_iter.next()
            except:
                neg_train_iter = iter(dataloader_neg)
                x2, y2 = neg_train_iter.next()
                # x3, y3 = neg_train_iter.next()
                # x4, y4 = neg_train_iter.next()
                # x5, y5 = neg_train_iter.next()

            ypred, features = model(x1.to(device, non_blocking=True))

            x_ = torch.cat((x1, x2), dim=0).to(device, non_blocking=True)
            image_new = snapmix(input=x_, features=features, target=y1.to(device, non_blocking=True), prob=1.0, beta=1.0,
                                cropsize=512, netname='regnet', model=model)
            for i in range(image_new.size(0)):
                image = image_new[i].permute(1, 2, 0).detach().cpu().numpy() * IMG_STD_ALL + IMG_MEAN_ALL
                imageio.imsave(sample_path + '//' + str(idx) + '_' + str(i) + '_1.png', image.astype('uint8'))

            # x_ = torch.cat((x1, x3), dim=0).to(device, non_blocking=True)
            # image_new = snapmix(input=x_, features=features, target=y1.to(device, non_blocking=True), prob=1.0,
            #                     beta=1.0,
            #                     cropsize=512, netname='regnet', model=model)
            # for i in range(image_new.size(0)):
            #     image = image_new[i].permute(1, 2, 0).detach().cpu().numpy() * IMG_STD_ALL + IMG_MEAN_ALL
            #     imageio.imsave(sample_path + '//' + str(idx) + '_' + str(i) + '_2.png', image.astype('uint8'))
            # # # #
            # x_ = torch.cat((x1, x4), dim=0).to(device, non_blocking=True)
            # image_new = snapmix(input=x_, features=features, target=y1.to(device, non_blocking=True), prob=1.0,
            #                     beta=1.0,
            #                     cropsize=512, netname='regnet', model=model)
            # for i in range(image_new.size(0)):
            #     image = image_new[i].permute(1, 2, 0).detach().cpu().numpy() * IMG_STD_ALL + IMG_MEAN_ALL
            #     imageio.imsave(sample_path + '//' + str(idx) + '_' + str(i) + '_3.png', image.astype('uint8'))
            # # #
            # x_ = torch.cat((x1, x5), dim=0).to(device, non_blocking=True)
            # image_new = snapmix(input=x_, features=features, target=y1.to(device, non_blocking=True), prob=1.0,
            #                     beta=1.0,
            #                     cropsize=512, netname='regnet', model=model)
            # for i in range(image_new.size(0)):
            #     image = image_new[i].permute(1, 2, 0).detach().cpu().numpy() * IMG_STD_ALL + IMG_MEAN_ALL
            #     imageio.imsave(sample_path + '//' + str(idx) + '_' + str(i) + '_4.png', image.astype('uint8'))
            idx = idx + 1

# def MIX_UP(net, dataloader_pos, dataloader_neg, device,alpha, sample_path,trainlist_pos, trainlist_mix_pos):
#     net.eval()
#     shutil.copy(trainlist_pos, trainlist_mix_pos)
#     list_sample = pd.read_csv(trainlist_mix_pos, sep=',', header=None)
#     Note = open(trainlist_mix_pos, mode='w')
#     for i in range(len(list_sample)):
#          Note.write('\n' + list_sample.iloc[i,0] + ',' + str(list_sample.iloc[i,1]) + ',' + str(1 - list_sample.iloc[i,1]) + ',1')
#     Note.close()
#     Note = open(trainlist_mix_pos, mode='a')
#     with torch.no_grad():
#         neg_train_iter = iter(dataloader_neg)  # negative samples
#         idx = 0
#         for x1, y1 in tqdm(dataloader_pos):
#             try:
#                 x2, y2 = neg_train_iter.next()
#             except:
#                 neg_train_iter = iter(dataloader_neg)
#                 x2, y2 = neg_train_iter.next()
#             # combine pos and negtor
#             x = torch.cat((x1, x2), dim=0).to(device, non_blocking=True) # N C H W
#             y_true = torch.cat((y1, y2), dim=0).to(device, non_blocking=True) # N H W
#
#             inputs, targets_a, targets_b, lam = mixup_data(x, y_true, alpha, True)
#             inputs, targets_a, targets_b = map(Variable, (inputs,
#                                                           targets_a, targets_b))
#             for i in range(inputs.size(0)):
#                 image = inputs[i].permute(1, 2, 0).detach().cpu().numpy() * IMG_STD_ALL + IMG_MEAN_ALL
#                 imageio.imsave(sample_path + '//' + str(idx) + '_' + str(i) + '_1.png', image.astype('uint8'))
#                 Note.write('\n' + sample_path + '//' + str(idx) + '_' + str(i) + '_1.png' + ',' + str(int(targets_a[i]))
#                            + ',' + str(int(targets_b[i])) + ',' + str(lam))
#             idx = idx + 1
#     Note.close()
def vtest_epoch(model, criterion, dataloader, device, epoch, classes):
    model.eval()
    acc_total = ClassificationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
   
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True)
            ypred,_ = model.forward(x)
            loss = criterion(ypred, y_true)

            ypred = ypred.argmax(axis=1)
            acc_total.addBatch(ypred, y_true)

            losses.update(loss.item(), x.size(0))
            oa = acc_total.OverallAccuracy()
            f1 = acc_total.F1score()
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, F1: {lv:.3f}, {fei:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, lv=f1[0], fei=f1[1]))
            pbar.update()
        pbar.close()
  
    oa = acc_total.OverallAccuracy()
    f1 = acc_total.F1score()
    return losses.avg, oa, f1

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, beta, cutmix_prob):
    r = np.random.rand(1)
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(x.size()[0]).cuda()
        target_a = y
        target_b = y[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, target_a, target_b, lam
    else:
        return x, y, y, 1
        # compute output
        # output = model(input)
        # loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def get_spm(input,target,cropsize,netname,model):

    imgsize = (cropsize,cropsize)
    bs = input.size(0)
    with torch.no_grad():
        output,fms = model(input)
        if 'inception' in netname:
            clsw = model.module.fc
        else:
            clsw = model.classification_head[3]
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(fms)
        poolfea = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea))
        logitlist = []
        for i in range(bs):
            logitlist.append(clslogit[i,target[i]])
        clslogit = torch.stack(logitlist)

        out = F.conv2d(fms, weight, bias=bias)

        outmaps = []
        for i in range(bs):
            evimap = out[i,target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
            outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()


    return outmaps,clslogit

def snapmix_data(input,target,prob,beta,cropsize,netname,model=None):

    r = np.random.rand(1)
    lam_a = torch.ones(input.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    if r < prob:
        wfmaps,_ = get_spm(input,target,cropsize,netname,model)
        bs = input.size(0)
        lam = np.random.beta(beta, beta)
        lam1 = np.random.beta(beta, beta)
        rand_index = torch.randperm(bs).cuda()
        wfmaps_b = wfmaps[rand_index,:,:]
        target_b = target[rand_index]

        same_label = target == target_b
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)

        area = (bby2-bby1)*(bbx2-bbx1)
        area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)

        if  area1 > 0 and  area>0:
            ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
            ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
            input[:, :, bbx1:bbx2, bby1:bby2] = ncont
            lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
            lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
            tmp = lam_a.clone()
            lam_a[same_label] += lam_b[same_label]
            lam_b[same_label] += tmp[same_label]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            lam_a[torch.isnan(lam_a)] = lam
            lam_b[torch.isnan(lam_b)] = 1-lam

    return input,target,target_b,lam_a.cuda(),lam_b.cuda()

def fftfreqnd(h, w=None, z=None):
    """ Get bin values for discrete fourier transform of size (h, w, z)
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    fz = fx = 0
    fy = np.fft.fftfreq(h)

    if w is not None:
        fy = np.expand_dims(fy, -1)

        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]

    if z is not None:
        fy = np.expand_dims(fy, -1)
        if z % 2 == 1:
            fz = np.fft.fftfreq(z)[:, None]
        else:
            fz = np.fft.fftfreq(z)[:, None]

    return np.sqrt(fx * fx + fy * fy + fz * fz)

def get_spectrum(freqs, decay_power, ch, h, w=0, z=0):
    """ Samples a fourier image with given size and frequencies decayed by decay power
    :param freqs: Bin values for the discrete fourier transform
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param ch: Number of channels for the resulting mask
    :param h: Required, first dimension size
    :param w: Optional, second dimension size
    :param z: Optional, third dimension size
    """
    scale = np.ones(1) / (np.maximum(freqs, np.array([1. / max(w, h, z)])) ** decay_power)

    param_size = [ch] + list(freqs.shape) + [2]
    param = np.random.randn(*param_size)

    scale = np.expand_dims(scale, -1)[None, :]

    return scale * param

def make_low_freq_image(decay, shape, ch=1):
    """ Sample a low frequency image from fourier space
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param ch: Number of channels for desired mask
    """
    freqs = fftfreqnd(*shape)
    spectrum = get_spectrum(freqs, decay, ch, *shape)#.reshape((1, *shape[:-1], -1))
    spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
    mask = np.real(np.fft.irfftn(spectrum, shape))

    if len(shape) == 1:
        mask = mask[:1, :shape[0]]
    if len(shape) == 2:
        mask = mask[:1, :shape[0], :shape[1]]
    if len(shape) == 3:
        mask = mask[:1, :shape[0], :shape[1], :shape[2]]

    mask = mask
    mask = (mask - mask.min())
    mask = mask / mask.max()
    return mask

def sample_lam(alpha, reformulate=False):
    """ Sample a lambda from symmetric beta distribution with given alpha
    :param alpha: Alpha value for beta distribution
    :param reformulate: If True, uses the reformulation of [1].
    """
    if reformulate:
        lam = beta.rvs(alpha+1, alpha)
    else:
        lam = beta.rvs(alpha, alpha)

    return lam

def binarise_mask(mask, lam, in_shape, max_soft=0.0):
    """ Binarises a given low frequency image such that it has mean lambda.
    :param mask: Low frequency image, usually the result of `make_low_freq_image`
    :param lam: Mean value of final mask
    :param in_shape: Shape of inputs
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :return:
    """
    idx = mask.reshape(-1).argsort()[::-1]
    mask = mask.reshape(-1)
    num = math.ceil(lam * mask.size) if random.random() > 0.5 else math.floor(lam * mask.size)

    eff_soft = max_soft
    if max_soft > lam or max_soft > (1-lam):
        eff_soft = min(lam, 1-lam)

    soft = int(mask.size * eff_soft)
    num_low = num - soft
    num_high = num + soft

    mask[idx[:num_high]] = 1
    mask[idx[num_low:]] = 0
    mask[idx[num_low:num_high]] = np.linspace(1, 0, (num_high - num_low))

    mask = mask.reshape((1, *in_shape))
    return mask

def sample_mask(alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """ Samples a mean lambda from beta distribution parametrised by alpha, creates a low frequency image and binarises
    it based on this lambda
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    """
    if isinstance(shape, int):
        shape = (shape,)

    # Choose lambda
    lam = sample_lam(alpha, reformulate)

    # Make mask, get mean / std
    mask = make_low_freq_image(decay_power, shape)
    mask = binarise_mask(mask, lam, shape, max_soft)

    return lam, mask

def sample_and_apply(x, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """
    :param x: Image batch on which to apply fmix of shape [b, c, shape*]
    :param alpha: Alpha value for beta distribution from which to sample mean of mask
    :param decay_power: Decay power for frequency decay prop 1/f**d
    :param shape: Shape of desired mask, list up to 3 dims
    :param max_soft: Softening value between 0 and 0.5 which smooths hard edges in the mask.
    :param reformulate: If True, uses the reformulation of [1].
    :return: mixed input, permutation indices, lambda value of mix,
    """
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    index = np.random.permutation(x.shape[0])

    x1, x2 = x * mask, x[index] * (1-mask)
    return x1+x2, index, lam

def fmix_data(x, y, decay_power=3, alpha=1, size=(512, 512), max_soft=0.0, reformulate=False):
    lam, mask = sample_mask(alpha, decay_power, size, max_soft, reformulate)
    index = torch.randperm(x.size(0)).to(x.device)
    mask = torch.from_numpy(mask).float().to(x.device)

    # Mix the images
    x1 = mask * x
    x2 = (1 - mask) * x[index]
    index = index
    lam = lam
    return x1 + x2, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lama, lamb):
    loss = lama * criterion(pred, y_a) + lamb * criterion(pred, y_b)
    return torch.mean(loss)

if __name__ == "__main__":
    main()