import torch
import numpy as np
import tifffile as tif
from osgeo import gdal
from metrics import SegmentationMetric, accprint

def arr2img(save_path, arr, img_width, img_height, adf_GeoTransform, im_Proj):
    # 保存为jpg格式
    # plt.imsave(save_path, arr)

    # 保存为TIF格式
    driver = gdal.GetDriverByName("GTiff")
    datasetnew = driver.Create(save_path, img_width, img_height, 1, gdal.GDT_Byte)
    datasetnew.SetGeoTransform(adf_GeoTransform)
    datasetnew.SetProjection(im_Proj)
    band = datasetnew.GetRasterBand(1)
    band.WriteArray(arr)
    datasetnew.FlushCache()  # Write to disk.必须有清除缓存

def main():
    file = r'D:\wwr\lvwang\Large_scale\Ge\foshan.tif'
    geo = gdal.Open(file)
    img_width = geo.RasterXSize
    img_height = geo.RasterYSize
    adf_GeoTransform = geo.GetGeoTransform()
    im_Proj = geo.GetProjection()

    cls_path = r'D:\wwr\Pred\Large_scale\Pred_GE_cls_foshan.tif'
    seg_path = r'D:\wwr\Pred\Large_scale\IRNet\Pred_IRNet_foshan_seg.tif'
    image_cls = tif.imread(cls_path)
    image_seg = tif.imread(seg_path)
    the =((np.max(image_seg) - np.min(image_seg)) / 2) + np.min(image_seg)
    final_thre = np.floor(the * 10) / 10
    print(the)
    print(final_thre)
    thre_cls = 0.01
    image_seg[image_cls < thre_cls] = 0
    result = np.zeros(image_seg.shape,dtype='uint8')
    result[image_seg > final_thre] = 1
    # arr2img(r'D:\wwr\Pred\Large_scale\SRIWS\result_SRIWS_foshan.tif', result, img_width, img_height, adf_GeoTransform, im_Proj)
    label_path = r'D:\wwr\lvwang\Large_scale\Ge\foshan_label_final.tif'
    label = tif.imread(label_path)
    acc_total = SegmentationMetric(numClass=2, device='cpu')
    acc_total.addBatch(torch.from_numpy(result), torch.from_numpy(label))
    accprint(acc_total)



if __name__ == '__main__':
    main()