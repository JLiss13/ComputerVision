#Proj1_2
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
from scipy import stats
import time
Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/Baudin_Liss_CE264Proj_1/'#Change this directory to the directory on your computer
Dir1=os.path.join(Dir,'Code/P1_1')
Dir_plots=os.path.join(Dir,'Deliverable/Images')
Dir2=os.path.join(Dir,'Photos/P1_2/Jpeg')
Array_txt_file="Proj1_2ArrayJ.txt"
plt.close('all')
start = time.time()
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
file = open(os.path.join(Dir,Array_txt_file), "w")
file.close()
filelist=os.listdir(Dir2)
filelist=filter(lambda k: '.jpeg' in k, filelist)
#Grab All metadata from the images in directory
M_Img_ArrayJ=['0','0','0','0','0','0','0','0']
for img in filelist:
    path=os.path.join(Dir2,img)
    PicInfoJ=get_jpeg_exif(path)
    image=cv2.imread(path,1)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    PicInfoJ[5]=str(np.mean(image[:,:,0]))
    PicInfoJ[6]=str(np.mean(image[:,:,1]))
    PicInfoJ[7]=str(np.mean(image[:,:,2]))
    print(PicInfoJ)
    M_Img_ArrayJ=np.vstack((M_Img_ArrayJ,PicInfoJ))
    np.savetxt(os.path.join(Dir2,Array_txt_file), M_Img_ArrayJ,fmt='%20.30s', delimiter=',')
        #plot histograms for each image
    plt.figure()
    color = ('r','g','b')
    for i,col in enumerate(color):
        histr = cv2.calcHist([image],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title(img)
    plt.show()
    plt.savefig(os.path.join(Dir_plots,'Histogram'+str(img)))
#IMG_6390.jpeg and IMG_6391.jpeg are the optimal exposure photos
end = time.time()
print(str(end - start)+" Seconds")
