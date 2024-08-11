import torch.nn.functional as F
import albumentations as A
import numpy as np
import torch
imgsize = 512
image_transform = A.Compose([
    # A.CenterCrop(width=imgsize, height=imgsize, always_apply=True),
    A.Flip(p=0.5),
    # A.RandomGridShuffle(grid=(2, 2), p=0.5),
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

def get_spm(features,target,cropsize,netname,model):

    imgsize = (cropsize,cropsize)
    bs = features.size(0)
    with torch.no_grad():
        # output,fms = model(input)
        if 'inception' in netname:
            clsw = model.module.fc
        else:
            # clsw = model.module.classifier
            clsw = model.classification_head[3]
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0),weight.size(1),1,1)
        fms = F.relu(features)
        poolfea = F.adaptive_avg_pool2d(fms,(1,1)).squeeze()
        clslogit = F.softmax(clsw.forward(poolfea),dim=1)
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
            # outmaps = F.interpolate(outmaps,imgsize,mode='bilinear',align_corners=False)

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()

    return outmaps,clslogit

def snapmix(input,features,target,prob,beta,cropsize,netname,grid_scale=32, model=None,top_k=None):

    r = np.random.rand(1)


    if r < prob:
        wfmaps,_ = get_spm(features,target,cropsize,netname,model)
        bs, _, att_size, _ = features.size()
        att_grid = att_size ** 2
        lam = np.random.beta(beta, beta)
        if top_k is None:
            top_k = min(max(att_grid // 4, int(att_grid * lam)), 3 * (att_grid // 4))
        _, att_idx = wfmaps.view(bs, att_grid).topk(top_k)
        att_idx = torch.cat([
            (att_idx // att_size).unsqueeze(1),
            (att_idx % att_size).unsqueeze(1), ], dim=1)
        mask = torch.zeros(bs, 1, att_size, att_size).cuda()
        for i in range(bs):
            mask[i, 0, att_idx[i, 0, :], att_idx[i, 1, :]] = 1
        mask = F.upsample(mask, scale_factor=grid_scale, mode="nearest")
        input = mask*input[:bs] + (1-mask)*input[bs:]
      

        for i in range(bs):
            transformed = image_transform(image=input[i].detach().cpu().numpy().transpose(1,2,0))
            input[i] = torch.from_numpy(transformed["image"].transpose(2,0,1)).float()

        return input.cuda()