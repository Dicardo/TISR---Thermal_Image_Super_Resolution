import cv2
import os
from glob import glob

lr_file = '/home/sba/sunzheng/ChasNet_x4/challengedataset/train/160_domo_lr_sift/'
hr_file = '/home/sba/sunzheng/ChasNet_x4/challengedataset/train/640_flir_hr_sift/'

rgb_file = lr_file
d_file = hr_file

crop_resize = 0.8


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def crop_image(img, point_w, w, point_h, h):
    crop_img = img[point_h:point_h + h, point_w:point_w + w]
    return crop_img


mkdir('/home/sba/sunzheng/ChasNet_x4/challengedataset/train/160_domo_lr_sift_crop')
mkdir('/home/sba/sunzheng/ChasNet_x4/challengedataset/train/640_flir_hr_sift_crop')

rgb_data = sorted(glob(os.path.join(rgb_file, "*.jpg")))
d_data = sorted(glob(os.path.join(d_file, "*.jpg")))

i = 0
# for i in range(0, 160, 40)
for filename_rgb in rgb_data:
    print(filename_rgb)
    img1 = cv2.imread(filename_rgb, 0)
    crop_rgb = crop_image(img1, int(160 * (1 - crop_resize) / 2), int(160 * crop_resize),
                          int(120 * (1 - crop_resize) / 2), int(120 * crop_resize))
    cv2.imwrite('/home/sba/sunzheng/ChasNet_x4/challengedataset/train/160_domo_lr_sift_crop/' + str(i) + '.jpg',
                crop_rgb)
    i += 1

i = 0
for filename_d in d_data:
    img2 = cv2.imread(filename_d, 0)
    crop_d = crop_image(img2, int(640 * (1 - crop_resize) / 2), int(640 * crop_resize),
                        int(480 * (1 - crop_resize) / 2), int(480 * crop_resize))
    cv2.imwrite('/home/sba/sunzheng/ChasNet_x4/challengedataset/train/640_flir_hr_sift_crop/' + str(i) + '.jpg', crop_d)
    i += 1
