import torch.nn.functional as F
import albumentations as A
import numpy as np
import torch

imgsize = 512
image_transform = A.Compose([
    A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
    A.Flip(p=0.5),
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
]
)
def rand_bbox(size, lam,center=False,attcen=None):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    elif len(size) == 2:
        W = size[0]
        H = size[1]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    if attcen is None:
        # uniform
        cx = 0
        cy = 0
        if W>0 and H>0:
            cx = np.random.randint(W)
            cy = np.random.randint(H)
        if center:
            cx = int(W/2)
            cy = int(H/2)
    else:
        cx = attcen[0]
        cy = attcen[1]

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_spm(features,cropsize,netname,model):

    imgsize = (cropsize,cropsize)
    bs = features.size(0)
    with torch.no_grad():
        # output,fms = model(input)
        if 'inception' in netname:
            clsw = model.module.fc
        else:
            # clsw = model.module.classifier
            clsw = model.segmentation_head[0]
        weight = clsw.weight.data
        bias = clsw.bias.data
        # weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(features)
        out = F.conv2d(fms, weight, bias=bias, padding=1)

        outmaps = []
        for i in range(bs):
            evimap = out[i,1]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)
        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0),1,outmaps.size(1),outmaps.size(2))
            # outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()


    return outmaps

def snapmix(input_forward, input_backward, SuperPix_forward, SuperPix_backward,
            update_forward, update_backward,features,prob,beta,cropsize,netname,grid_scale=32, model=None,top_k=None):

    r = np.random.rand(1)
    # lam_a = torch.ones(input.size(0))
    # lam_b = 1 - lam_a
    # target_b = target.clone()

    if r < prob:
        wfmaps = get_spm(features,cropsize,netname,model)
        bs, att_size, _ = wfmaps.size()
        att_grid = att_size ** 2
        lam = np.random.beta(beta, beta)
        if top_k is None:
            top_k = min(max(att_grid // 4, int(att_grid * lam)), 3 * (att_grid // 4))
        # features = features.mean(1)
        _, att_idx = wfmaps.view(bs, att_grid).topk(top_k)
        att_idx = torch.cat([
            (att_idx // att_size).unsqueeze(1),
            (att_idx % att_size).unsqueeze(1), ], dim=1)
        mask_snap = torch.zeros(bs, 1, att_size, att_size).cuda()
        for i in range(bs):
            mask_snap[i, 0, att_idx[i, 0, :], att_idx[i, 1, :]] = 1
            SuperPix_backward[i] = SuperPix_backward[i] + SuperPix_forward[i].max() + 1
            # mask_backward[i, mask[i].squeeze() == 1] = mask_forward[i, mask[i].squeeze() == 1]
            # update_backward[i, mask[i].squeeze() == 1] = update_forward[i, mask[i].squeeze() == 1]
            # SuperPix_backward[i, mask[i].squeeze() == 1] = SuperPix_forward[i, mask[i].squeeze() == 1]
        # mask_snap = F.upsample(mask_snap, scale_factor=grid_scale, mode="nearest").long()
        images = mask_snap*input_forward + (1-mask_snap)*input_backward
        SuperPix = mask_snap.squeeze()*SuperPix_forward + (1-mask_snap.squeeze())*SuperPix_backward
        # mask = mask_snap.squeeze()*mask_forward + (1-mask_snap.squeeze())*mask_backward
        update = mask_snap.squeeze()*update_forward + (1-mask_snap.squeeze())*update_backward

        for i in range(bs):
            masks = []
            masks.append(update[i].detach().cpu().numpy())
            masks.append(SuperPix[i].detach().cpu().numpy())
            transformed = image_transform(image=images[i].detach().cpu().numpy().transpose(1,2,0), masks=masks)
            images[i] = torch.from_numpy(transformed["image"].transpose(2,0,1)).float()
            update[i] = torch.from_numpy(transformed["masks"][0]).long()
            SuperPix[i] = torch.from_numpy(transformed["masks"][1]).long()
        # lam1 = np.random.beta(beta, beta)
        # # rand_index = torch.randperm(bs).cuda()
        # rand_index = torch.tensor(list(range(bs - 1, -1, -1))).cuda()
        # # wfmaps_b = wfmaps[rand_index,:,:]
        # # target_b = target[rand_index]
        #
        # # same_label = target == target_b
        # bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        # bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(input.size(), lam1)
        #
        # area = (bby2-bby1)*(bbx2-bbx1)
        # area1 = (bby2_1-bby1_1)*(bbx2_1-bbx1_1)
        #
        # if area1 > 0 and  area>0:
        #     ncont = input[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
        #     ncont = F.interpolate(ncont, size=(bbx2-bbx1,bby2-bby1), mode='bilinear', align_corners=True)
        #     input[:, :, bbx1:bbx2, bby1:bby2] = ncont
        #     # lam_a = 1 - wfmaps[:,bbx1:bbx2,bby1:bby2].sum(2).sum(1)/(wfmaps.sum(2).sum(1)+1e-8)
        #     # lam_b = wfmaps_b[:,bbx1_1:bbx2_1,bby1_1:bby2_1].sum(2).sum(1)/(wfmaps_b.sum(2).sum(1)+1e-8)
        #     # tmp = lam_a.clone()
        #     # lam_a[same_label] += lam_b[same_label]
        #     # lam_b[same_label] += tmp[same_label]
        #     # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        #     # lam_a[torch.isnan(lam_a)] = lam
        #     # lam_b[torch.isnan(lam_b)] = 1-lam
        return images.cuda(),update.cuda().long(),SuperPix.cuda().long()