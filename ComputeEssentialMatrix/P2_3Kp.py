import os
import ProjImgLib as PL
from matplotlib import pyplot as plt
import numpy as np
import cv2

Dir = os.getcwd()
print('Working Directory is: ' + Dir)
PhotoDir = os.path.split(Dir)
PhotoDir = os.path.split(PhotoDir[0])
PhotoDir = os.path.join(PhotoDir[0], 'Photos')
print('Photo directory is: ' + PhotoDir)

files=os.listdir(PhotoDir)
files=list(filter(lambda k: 'Scene_Undis&Cal_' in k, files))
files=list(filter(lambda k: '.JPG' in k, files))

# Find all unique points on the image using SURF feature detector
for imgname in files:
    print(imgname)
    img = cv2.imread(os.path.join(PhotoDir,imgname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1=img.copy()
    gray= cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    #kp,des,img=PL.SIFT_Det(PhotoDir,os.path.join(PhotoDir,imgname))
    kp,des,img2=PL.Surf_Det(PhotoDir, img, gray, imgname,9000)
    KeyMat=PL.KeyPoints(PhotoDir,imgname,kp)
    plt.figure()
    plt.title(imgname.replace('.JPG','_')+'with keypoints')
    KeyImg=cv2.drawKeypoints(img,kp,img,color=(255,0,0))
    plt.imshow(KeyImg)
