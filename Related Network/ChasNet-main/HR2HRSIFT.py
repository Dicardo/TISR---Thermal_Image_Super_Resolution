import os
import cv2

file_pathname = r"/home/sba/sunzheng/ChasNet_x4/challengedataset/train/160_domo_lr_sift"
file_dirname = r"/home/sba/sunzheng/ChasNet_x4/challengedataset/train"

path_list = os.listdir(file_pathname)
path_list.sort()

for filename in path_list:
    img1 = cv2.imread(file_dirname + '/640_flir_hr/' + filename)
    cv2.imwrite('/home/sba/sunzheng/ChasNet_x4/challengedataset/train/640_flir_hr_sift'+"/" + filename, img1)