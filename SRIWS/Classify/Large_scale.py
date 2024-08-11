import os
import time
import math
import torch
import random
import numpy as np
import tifffile as tif
from osgeo import gdal
from model import ClassificationModel

# for google img, RGB
IMG_MEAN_ALL_Ge = np.array([98.3519, 96.9567, 95.5713])
IMG_STD_ALL_Ge = np.array([52.7343, 45.8798, 44.3465])
# used for high resolution images 1 m
IMG_MEAN_ALL_Gf = np.array([495.742416, 379.526141, 322.051960, 276.685233])
IMG_STD_ALL_Gf = np.array([129.443616, 116.345968, 119.554353, 100.763886])

def predict_whole_image_cls(model, image, mask, device, grid=256):
    '''
    image: n,r,c,b  where n = 1
    model: FCN
    不重叠的预测
    update: 增加了分割图的输出
    '''
    #grid=400
    n,b,r,c = image.shape
    rows=math.ceil(r/grid)*grid
    cols=math.ceil(c/grid)*grid
    image_= np.pad(image,((0,0), (0,0), (0,rows-r), (0,cols-c)),'symmetric')
    mask_= np.pad(mask,((0,rows-r), (0,cols-c)),'constant')
    weight = np.ones((rows, cols))
    res = np.zeros((rows, cols), dtype=np.uint8)
    num_patch= len(range(0,rows,grid))*len(range(0,cols,grid))
    print('num of patch is',num_patch)
    k=0
    for i in range(0,rows, grid):
        for j in range(0, cols, grid):
            patch = image_[0:,0:,i:i+grid,j:j+grid]
            if np.max(mask_[i:i+grid, j:j+grid].flatten())<=10e-8: # use max
                continue
            start=time.time()
            patch = torch.from_numpy(patch).float()
            with torch.no_grad():
                pred = model(patch.to(device)).squeeze(0)
            pred = pred.argmax(0).cpu().numpy()
            res[i:i+grid,j:j+grid] = pred+1 # 1,2,3
            end=time.time()
            k=k+1
            if k % 500 ==0:
                print('patch [%d/%d] time elapse:%.3f'%(k,num_patch,(end-start)))
    res = res[0:r,0:c].astype(np.uint8)
    return res
#以2048为一个patch, 重叠64个像素即可
def predict_whole_image_over_cls(model, image,device, num_class=1, grid=512, stride=256):
    '''
    image: n,r,c,b  where n = 1
    model: FCN
    重叠的预测
    '''
    n,b,r,c = image.shape
    rows= math.ceil((r-grid)/(stride))*stride+grid
    cols= math.ceil((c-grid)/(stride))*stride+grid
#     rows=math.ceil(rows)
#     cols=math.ceil(cols)
    print('rows is {}, cols is {}'.format(rows,cols))
    image_= np.pad(image,((0,0),(0,0),(0,rows-r), (0,cols-c), ),'symmetric')
    weight = np.ones((rows, cols))
    res = np.zeros((num_class, rows, cols),dtype=np.float32)
    num_patch= len(range(0,rows,stride))*len(range(0,cols,stride))
    print('num of patch is',num_patch)
    k=0
    for i in range(0,rows, stride):
        for j in range(0, cols, stride):
            start=time.time()
            patch = image_[0:,0:,i:i+grid,j:j+grid]
            patch = torch.from_numpy(patch).float()
            with torch.no_grad():
                pred,_ = model(patch.to(device))
                pred = torch.softmax(pred, dim=1).squeeze(0) # 1 C
            pred = pred[0].cpu().numpy() # for lvwang, probability
            res[:, i:i+grid,j:j+grid] += pred # should be +=
            weight[i:i+grid,j:j+grid] += 1
            end=time.time()
            k=k+1
            if k % 500 ==0:
                print('patch [%d/%d] time elapse:%.3f'%(k,num_patch,(end-start)))
                #tif.imsave(os.path.join(ipath,'height{}_{}.tif'.format(i,j)),pred,dtype=np.float32)
    res= res/weight
    # res=np.argmax(res, axis=0)
    res = res[:, 0:r,0:c].astype(np.float32)
    res = np.squeeze(res)
    return res


def arr2img(save_path, arr, img_width, img_height, adf_GeoTransform, im_Proj):
    # 保存为jpg格式
    # plt.imsave(save_path, arr)

    # 保存为TIF格式
    driver = gdal.GetDriverByName("GTiff")
    datasetnew = driver.Create(save_path, img_width, img_height, 1, gdal.GDT_Float32)
    datasetnew.SetGeoTransform(adf_GeoTransform)
    datasetnew.SetProjection(im_Proj)
    band = datasetnew.GetRasterBand(1)
    band.WriteArray(arr)
    datasetnew.FlushCache()  # Write to disk.必须有清除缓存

def main():
    file = r'D:\wwr\lvwang\Large_scale\Ge\xiaogan.tif'
    geo = gdal.Open(file)
    img_width = geo.RasterXSize
    img_height = geo.RasterYSize
    adf_GeoTransform = geo.GetGeoTransform()
    im_Proj = geo.GetProjection()

    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # Ge数据集
    nchannels = 3
    classes = 2
    device = 'cuda'
    # 分类网络
    model = ClassificationModel(encoder_name="timm-regnety_040", encoder_weights="imagenet",
                              in_channels=nchannels, classes=classes).to(device)
    logdir_cls = r'Z:\弱监督建设用地实验\Classify\runs\ge\Classify_CAM_Atten_MIX 1'
    # print the model
    resume = os.path.join(logdir_cls, 'model_best.tar')
    final_path = 'D:\wwr\Pred\Large_scale\Pred_GE_cls_xiaogan.tif'
    try:
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> success '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    except:
        print("resume fails")
    model.eval()

    mux = tif.imread(file)[:, :, :3]
    mux = (mux - IMG_MEAN_ALL_Ge) / IMG_STD_ALL_Ge  # normalized to [0,1]
    mux = np.expand_dims(mux.transpose(2, 0, 1), axis=0)
    print(mux.shape)
    mask = np.ones((mux.shape[2], mux.shape[3]), 'uint8')

    res_cls = predict_whole_image_over_cls(model, mux, device, num_class=1, grid=512, stride=256)
    tif.imwrite(final_path, res_cls)
    print('get')
    arr2img(final_path, res_cls, img_width, img_height, adf_GeoTransform, im_Proj)

    #GF2数据集
    # nchannels = 4
    # classes = 2
    # device = 'cuda'
    # # 分类网络
    # model = ClassificationModel(encoder_name="timm-regnety_040", encoder_weights="imagenet",
    #                             in_channels=nchannels, classes=classes).to(device)
    # logdir_cls = r'Z:\弱监督建设用地实验\Classify\runs\gf2\Classify_CAM_Atten_MIX1'
    # # print the model
    # resume = os.path.join(logdir_cls, 'model_best.tar')
    # final_path = 'D:/wwr/lvwang/Pred_SRIWS_GF_cls.tif'
    # try:
    #     print("=> loading checkpoint '{}'".format(resume))
    #     checkpoint = torch.load(resume)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     # optimizer.load_state_dict(checkpoint['optimizer'])
    #     print("=> success '{}' (epoch {})"
    #           .format(resume, checkpoint['epoch']))
    # except:
    #     print("resume fails")
    # model.eval()
    #
    # file = r'D:\wwr\lvwang\xiaogan.tif'
    # mux = tif.imread(file)
    # mux = (mux - IMG_MEAN_ALL_Gf) / IMG_STD_ALL_Gf  # normalized to [0,1]
    # mux = np.expand_dims(mux.transpose(2, 0, 1), axis=0)
    # print(mux.shape)
    # mask = np.ones((mux.shape[2], mux.shape[3]), 'uint8')
    #
    # res_cls = predict_whole_image_over_cls(model, mux, device, num_class=1, grid=512, stride=256)
    # tif.imwrite(final_path, res_cls)
    # print('get')

if __name__ == '__main__':
    main()