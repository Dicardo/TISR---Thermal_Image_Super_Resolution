#导入 opencv
import cv2

from skimage import io,transform,color
import numpy as np

def convert_gray(f, **args):  # 图片处理与格式化的函数
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray


datapath = r'C:\Users\Dicardo\Desktop\640_flir_hr'  # 图片所在的路径
str = datapath + '/*.jpg'  # 识别.jpg的图像
coll = io.ImageCollection(str, load_func=convert_gray)  # 批处理
for i in range(len(coll)):
    io.imsave(r'C:\Users\Dicardo\Desktop\train_depth_hr\\' + np.str(i) + '.jpg', coll[i])  # 保存图片在d:/input_image4