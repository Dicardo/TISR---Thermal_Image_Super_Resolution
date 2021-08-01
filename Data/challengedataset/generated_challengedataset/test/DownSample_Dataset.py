import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from DownSample import gaussian_noise2


def PicDown(source_Dir, new_Dir,scale_factor=2):
    '''
    :param source_Dir: 要处理的文件夹路径
    :param new_Dir: 保存的路径
    :param scale_factor: 缩放倍数
    :return:无
    '''
    # 判断是否存在保存路径，不存在则新建
    if not os.path.exists(new_Dir):
        os.makedirs(new_Dir)

    # 处理图片
    for filename in os.listdir(source_Dir):
        # 打印文件名
        print(filename)
        # 读取图片
        img = cv2.imread(os.path.join(source_Dir, filename))
        # 为图片添加高斯噪声
        img_noise = gaussian_noise2(img)
        # 将图片下采样
        if scale_factor == 2: # 缩放2倍
            img_down = cv2.pyrDown(img_noise)
        elif scale_factor == 4: # 缩放4倍
            img_down = cv2.pyrDown(img_noise)
            img_down = cv2.pyrDown(img_down)

        # 保存图片
        cv2.imwrite(os.path.join(new_Dir, filename), img_down)
        print('处理成功')

# PicDown('./640_flir_hr','./HR_down2_MR/',scale_factor=2)
PicDown('/home/sba/sunzheng/Thermal_UNet/challengedataset/test/640_flir_hr','/home/sba/sunzheng/Thermal_UNet/challengedataset/test/HR_down2_MR',scale_factor=2)
