1. Classify文件夹：Classify_SnapMix_XX.py 分类
2. Classify文件夹：CAM 生产CAM图
3. Matlab：从CAM图中生产初始标签
4. pytorch_superpixpool文件夹：generate_unsupervised_segmentation_v1.py 生产超像素图
5. pytorch_superpixpool文件夹：train_final_XX.py 分割
运行环境：python 3.7 + CUDA 10.2 + torch1.10.0
需安装超像素池化层python库