#Proj1_2
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
start = time.time()
plt.close('all')
Dir_plots='/Users/Jaliss/Dropbox/UCSC/CMPE_264/CE264Proj_1/Deliverable/Images'
Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/Baudin_Liss_CE264Proj_1/'#Change this directory to the directory on your computer
Dir2=os.path.join(Dir,'Photos/P1_2/Jpeg')
Dir3=os.path.join(Dir,'Photos/P1_3/Jpeg')
Array_txt_file="Proj1_2ArrayJ.txt"
fname=os.path.join(Dir2,Array_txt_file)
M_Img_ArrayJ=np.genfromtxt(fname, delimiter=',', skip_header=0)
a1=10.0 # 10 times exposure
a2=20.0 # 20 times exposure
def HDRMin(Image1,Image2,Image3,a1,a2,Dir): # Zero out every pixel in the image that was 255
    Image1=Image1.copy()
    Image2=Image2.copy()
    Image3=Image3.copy()
    HDRshape=np.shape(Image1)
    HDR=np.zeros((HDRshape[0],HDRshape[1],HDRshape[2]))
    for i in range(0,HDRshape[0]-1):
        for j in range(0,HDRshape[1]-1):
            if np.min(Image1[i,j,:]) <= 255.0/a1:
                HDR[i,j,0]=Image2[i,j,0]
                HDR[i,j,1]=Image2[i,j,1]
                HDR[i,j,2]=Image2[i,j,2]
            elif np.min(Image1[i,j,:]) <= 255.0/a2:
                HDR[i,j,0]=Image3[i,j,0]
                HDR[i,j,1]=Image3[i,j,1]
                HDR[i,j,2]=Image3[i,j,2]
            else:
                HDR[i,j,0]=Image1[i,j,0]
                HDR[i,j,1]=Image1[i,j,1]
                HDR[i,j,2]=Image1[i,j,2]
    HDR=np.uint8(HDR)
    Dir=os.path.join(Dir,'Photos/P1_3/Jpeg')
    cv2.imwrite(os.path.join(Dir3,'HDR_ofStack_ImagesSimple.jpeg'),HDR)

    return HDR
def HDRAvg (Image1,Image2,Image3,Dir):
    Image1=Image1.copy()
    Image2=Image2.copy()
    Image3=Image3.copy()
    HDRshape=np.shape(Image1)
    HDR1=np.zeros((HDRshape[0],HDRshape[1],HDRshape[2]))
    for i in range(0,HDRshape[0]-1):
            for j in range(0,HDRshape[1]-1):
                if np.min(Image1[i,j,:]) <= 254 and np.min(Image2[i,j,:])*a1 <= 254 and np.min(Image3[i,j,:])*a2 <= 254:
                    HDR1[i,j,0]=(Image1[i,j,0]+Image2[i,j,0]+Image3[i,j,0])/3.0
                    HDR1[i,j,1]=(Image1[i,j,1]+Image2[i,j,1]+Image3[i,j,1])/3.0
                    HDR1[i,j,2]=(Image1[i,j,2]+Image2[i,j,2]+Image3[i,j,2])/3.0
                elif np.min(Image1[i,j,:]) <= 254 and np.min(Image2[i,j,:])*a1 <= 254 and np.min(Image3[i,j,:])*a2 >= 255:
                    HDR1[i,j,0]=(Image1[i,j,0]+Image2[i,j,0])/2.0
                    HDR1[i,j,1]=(Image1[i,j,1]+Image2[i,j,1])/2.0
                    HDR1[i,j,2]=(Image1[i,j,2]+Image2[i,j,2])/2.0
                else:
                    HDR1[i,j,0]=Image1[i,j,0]
                    HDR1[i,j,1]=Image1[i,j,1]
                    HDR1[i,j,2]=Image1[i,j,2]
    HDR1=np.uint8(HDR1)
    cv2.imwrite(os.path.join(Dir3,'HDR_ofStack_ImagesAvg.jpeg'),HDR1)
    return HDR1
def HDRWAvg (Image1,Image2,Image3,Dir,a1,a2,M_Img_ArrayJ):
    Image1=Image1.copy()
    Image2=Image2.copy()
    Image3=Image3.copy()
    HDRshape=np.shape(Image1)
    HDR2=np.zeros((HDRshape[0],HDRshape[1],HDRshape[2]))
    E1=float(M_Img_ArrayJ[1,6])
    E1T=E1*float(M_Img_ArrayJ[1,1])
    E2=float(M_Img_ArrayJ[11,6])
    E2T=E2*float(M_Img_ArrayJ[11,1])/pow(a1,2)
    E3=float(M_Img_ArrayJ[14,6])
    E3T=E3*float(M_Img_ArrayJ[14,1])/pow(a2,2)
    CW1=E1T/(E1T+E2T+E3T)
    W1=CW1 # You want to take the complement to reduce the effects of noisy data
    print("W1 "+ str(W1))
    CW2=E2T/(E1T+E2T+E3T)
    W2=CW2 # You want to take the complement to reduce the effects of noisy data
    print("W2 "+ str(W2))
    CW3=E3T/(E1T+E2T+E3T)
    W3=CW3 # You want to take the complement to reduce the effects of noisy data
    print("W3 "+ str(W3))
    print("Total "+str(W1+W2+W3))
    for i in range(0,HDRshape[0]-1):
        for j in range(0,HDRshape[1]-1):
            if np.min(Image1[i,j,:]) <= 254 and np.min(Image2[i,j,:])*a1 <= 254 and np.min(Image3[i,j,:])*a2 <= 254:
                HDR2[i,j,0]=(W1*Image1[i,j,0]+W2*Image2[i,j,0]+W3*Image3[i,j,0])
                HDR2[i,j,1]=(W1*Image1[i,j,1]+W2*Image2[i,j,1]+W3*Image3[i,j,1])
                HDR2[i,j,2]=(W1*Image1[i,j,2]+W2*Image2[i,j,2]+W3*Image3[i,j,2])
            elif np.min(Image1[i,j,:]) <= 254 and np.min(Image2[i,j,:])*a1 <= 254 and np.min(Image3[i,j,:])*a2 >= 255:
                HDR2[i,j,0]=(W1*Image1[i,j,0]+W2*Image2[i,j,0])
                HDR2[i,j,1]=(W1*Image1[i,j,1]+W2*Image2[i,j,1])
                HDR2[i,j,2]=(W1*Image1[i,j,2]+W2*Image2[i,j,2])
            else:
                HDR2[i,j,0]=Image1[i,j,0]
                HDR2[i,j,1]=Image1[i,j,1]
                HDR2[i,j,2]=Image1[i,j,2]
    HDR2=np.uint8(HDR2)
    cv2.imwrite(os.path.join(Dir3,'HDR_ofStack_ImagesWAvg.jpeg'),HDR2)
    return HDR2
def crop(img):
    Shape=np.shape(img)
    img=img[int(Shape[0]/2)-200:int(Shape[0]/2)+500,0:500,:]
    return img

img='Linearization_of_IMG_6390.jpeg'
path=os.path.join(Dir3,img)
ImageLin1=cv2.imread(path,1)
ImageLin1=crop(ImageLin1)

img='Linearization_of_IMG_6400.jpeg'
path=os.path.join(Dir3,img)
ImageLin2=cv2.imread(path,1)
ImageLin2=crop(ImageLin2)

img='Linearization_of_IMG_6403.jpeg'
path=os.path.join(Dir3,img)
ImageLin3=cv2.imread(path,1)
ImageLin3=crop(ImageLin3)

#Divide the pixels of Linearized Image 2 and 3 to convert these values to equivalent exposure times
ImageLin2=ImageLin2.copy()
ImageLin2=ImageLin2/10.0
ImageLin3=ImageLin3.copy()
ImageLin3=ImageLin3/20.0

#Write Plot HDR Photos
'''
print("HDR Simple Start")
ImageHDR3S=HDRMin(ImageLin1,ImageLin2,ImageLin3,a1,a2, Dir) # a1=10 a2=20
ImageHDR3S = cv2.cvtColor(ImageHDR3S,cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(ImageHDR3S)
plt.show()
plt.title("Image After HDR Simple with 3 Photos")
print("HDR Simple End")

print("HDR Average Start")
ImageHDR3A=HDRAvg(ImageLin1,ImageLin2,ImageLin3,Dir)
ImageHDR3A = cv2.cvtColor(ImageHDR3A,cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(ImageHDR3A)
plt.show()
plt.title("Image After HDR Avg with 3 Photos")
print("HDR Average End")
'''
print("HDR Weighted Average Start")
ImageHDR3WA=HDRWAvg(ImageLin1,ImageLin2, ImageLin3,Dir,a1,a2,M_Img_ArrayJ) # a1=10 a2=20
ImageHDR3WA = cv2.cvtColor(ImageHDR3WA,cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(ImageHDR3WA)
plt.show()
plt.title("Image After HDR Weighted Avg with 3 Photos")
print("HDR Weighted Average End")

end = time.time()
print(str(end - start)+" Seconds")
