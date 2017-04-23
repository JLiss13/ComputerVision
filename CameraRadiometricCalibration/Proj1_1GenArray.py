#Project 1_1
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS


Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/Baudin_Liss_CE264Proj_1/' #Change this directory
Dir1=os.path.join(Dir,'Code/P1_1')
plt.close("all")

#Write to a txt file
Dir2=os.path.join(Dir,'Photos/P1_1/Jpeg')
Array_txt_file="Proj1_1ArrayJ.txt"
file = open(os.path.join(Dir,Array_txt_file), "w")
file.close()
def get_jpeg_exif(fn): # Grab metadata from the photos
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        ret[decoded] = value
    fn=os.path.split(fn)
    fn=fn[1]
    Name=fn
    T=ret['ExposureTime']
    T=str(float(T[0])/float(T[1]))
    G=str(float(ret['ISOSpeedRatings']))
    W=str(ret['ExifImageWidth'])
    H=str(ret['ExifImageHeight'])
    B='0'
    G='0'
    R='0'
    PicInfo=[Name,T,G,W,H,R,G,B]
    return PicInfo
# Grabbing Photos and Metadata from JPEG Photos
filelist=os.listdir(Dir2)
filelist=filter(lambda k: '.JPG' in k, filelist)
M_Img_ArrayJ=['0','0','0','0','0','0','0','0']
for img in filelist:
    path=os.path.join(Dir2,img)
    PicInfoJ=get_jpeg_exif(path) # Do not change the name of the jpeg_exif attribute
    image=cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=np.array(image)
    crop_image=image[int(PicInfoJ[3])/2-5:int(PicInfoJ[3])/2+5,int(PicInfoJ[4])/2-5:int(PicInfoJ[4])/2+5]
    PicInfoJ[5]=str(np.mean(crop_image[:,:,0]))
    PicInfoJ[6]=str(np.mean(crop_image[:,:,1]))
    PicInfoJ[7]=str(np.mean(crop_image[:,:,2]))
    M_Img_ArrayJ=np.vstack((M_Img_ArrayJ,PicInfoJ))
    np.savetxt(os.path.join(Dir2,Array_txt_file), M_Img_ArrayJ,fmt='%20.30s', delimiter=',')
    PicInfoJtxt=PicInfoJ[0]+','+PicInfoJ[1]+','+PicInfoJ[2]+','+PicInfoJ[3]+','+PicInfoJ[4]+','+ \
    PicInfoJ[5]+','+PicInfoJ[6]+','+PicInfoJ[7]
    print(PicInfoJtxt)
plt.figure()
plt.imshow(crop_image, interpolation='none')
plt.show()

#PrintToCSVReport(os.path.join(Dir,Array_txt_file),M_Img_ArrayJ) #Grab new lines and add new lines
Dir3=os.path.join(Dir,'Photos/P1_1/Raw')

#Write to a txt file
Array_txt_file="Proj1_1ArrayR.txt"
file = open(os.path.join(Dir,Array_txt_file), "w")
file.close()
'''
# Grabbing Photos and Metadata from Raw Photos
filelist=os.listdir(Dir)
filelist=filter(lambda k: '.CR2' in k, filelist)
M_Img_ArrayR=['0','0','0','0','0','0','0','0']
for img in filelist:
    PicInfoR=get_raw_exif(path) # Do not change the name of the  raw_exif attribute
    path=os.path.join(Dir,img)
    image= rawpy.imread(path)
    image= image.postprocess()
    crop_image=image[int(PicInfoR[3])/2-50:int(PicInfoR[3])/2+50,int(PicInfoR[4])/2-50:int(PicInfoR[4])/2+50]
    plt.figure()
    plt.imshow(image, interpolation='none')
    plt.show()
    PicInfoR[5]=str(np.mean(crop_image[0:9,0:9,0]))
    PicInfoR[6]=str(np.mean(crop_image[0:9,0:9,1]))
    PicInfoR[7]=str(np.mean(crop_image[0:9,0:9,2]))
    M_Img_ArrayR=np.vstack((M_Img_ArrayR,PicInfoR))
    np.savetxt(os.path.join(Dir,Array_txt_file), M_Img_ArrayR,fmt='%20.30s', delimiter=',')
    PicInfoRtxt=PicInfoR[0]+','+PicInfoR[1]+','+PicInfoR[2]+','+PicInfoR[3]+','+PicInfoR[4]+','+ \
    PicInfoR[5]+','+PicInfoR[6]+','+PicInfoR[7]
    print(PicInfoRtxt)
'''
