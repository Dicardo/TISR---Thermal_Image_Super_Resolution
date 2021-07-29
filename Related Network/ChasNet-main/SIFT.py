import numpy as np
import cv2
import os

def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des

'''
def siftImageAlignment(img1,img2):
   _,kp1,des1 = sift_kp(img1)
   _,kp2,des2 = sift_kp(img2)
   goodMatch = get_good_match(des1,des2)
   print(len(goodMatch))
   if len(goodMatch) > 4:
       ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ransacReprojThreshold = 4
       H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
       imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
   return imgOut,H,status
'''

file_pathname = r"/home/sba/sunzheng/ChasNet_x4/challengedataset/train/160_domo_lr"
file_dirname = r"/home/sba/sunzheng/ChasNet_x4/challengedataset/train"

'''
filename = '0071.jpg'
img1 = cv2.imread(file_dirname + '/160_domo_lr/' + filename)
print(file_dirname + '/160_domo_lr/' + filename)
img2 = cv2.imread(file_dirname + '/640_flir_hr/' + filename)
print(file_dirname + '/640_flir_hr/' + filename)
result, _, _ = siftImageAlignment(img1, img2)
cv2.imwrite('/home/sba/sunzheng/ChasNet_x4/challengedataset/train/160_domo_lr_sift'+"/" + filename, result)
'''

path_list = os.listdir(file_pathname)
path_list.sort()

for filename in path_list:
    img1 = cv2.imread(file_dirname + '/160_domo_lr/' + filename)
    img2 = cv2.imread(file_dirname + '/640_flir_hr/' + filename)
    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1,des2)
    if len(goodMatch) > 4:
        ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
        result = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        continue
    cv2.imwrite('/home/sba/sunzheng/ChasNet_x4/challengedataset/train/160_domo_lr_sift'+"/" + filename, result)


