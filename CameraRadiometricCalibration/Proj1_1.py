#Project 1_1
import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
import rawpy
Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/CE264Proj_1/Code/P1_1'
os.chdir(Dir)
from ProjImgLib import get_raw_exif, get_jpeg_exif
plt.close("all")

# Grabbing Photos and Metadata from Photos
Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/CE264Proj_1/Photos/P1_1/Jpeg'
filelist=os.listdir(Dir)
filelist=filter(lambda k: '.JPG' in k, filelist)
M_Img_ArrayJ=['0','0','0','0','0','0','0','0']
for img in filelist:
    path=os.path.join(Dir,img)
    PicInfoJ=get_jpeg_exif(path) # Do not change the name of the jpeg_exif attribute
    M_Img_ArrayJ=np.vstack((M_Img_ArrayJ,PicInfoJ))
    image=cv2.imread(path)
    image=np.array(image)
    crop_image=image[PicInfoJ[3]/2-5:PicInfoJ[3]/2+5,PicInfoJ[4]/2-5:PicInfoJ[4]/2+5]
    PicInfoJ[5]=str(np.mean(crop_image[:,:,0]))
    PicInfoJ[6]=str(np.mean(crop_image[:,:,1]))
    PicInfoJ[7]=str(np.mean(crop_image[:,:,2]))
    print(PicInfoJ)
plt.figure()
plt.imshow(crop_image, interpolation='none')
plt.show() 
Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/CE264Proj_1/Photos/P1_1/Raw'
#img=raw_input('Which image file?')
filelist=os.listdir(Dir)
filelist=filter(lambda k: '.CR2' in k, filelist)
M_Img_ArrayR=['0','0','0','0','0','0','0','0']
for img in filelist:
    PicInfoR=get_raw_exif(path) # Do not change the name of the  raw_exif attribute
    M_Img_ArrayR=np.vstack((M_Img_ArrayR,PicInfoR))
    path=os.path.join(Dir,img)
    image= rawpy.imread(path)
    image= image.postprocess()
    crop_image=image[int(PicInfoJ[3])/2-5:int(PicInfoJ[3])/2+5,int(PicInfoJ[4])/2-5:int(PicInfoJ[4])/2+5]
    PicInfoR[5]=str(np.mean(crop_image[0:9,0:9,0]))
    PicInfoR[6]=str(np.mean(crop_image[0:9,0:9,1]))
    PicInfoR[7]=str(np.mean(crop_image[0:9,0:9,2]))
    print(PicInfoR)
plt.figure()
plt.imshow(crop_image, interpolation='none')
plt.show()

# Plotting Brightness vs. Exposure for Jpeg and Raw Files
plt.figure()
plt.plot(float(PicInfoJ[:,1]),float(PicInfoJ[:,5])) #Red
plt.show()