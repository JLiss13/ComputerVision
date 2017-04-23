#Proj1_2
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
start = time.time()
plt.close('all')
Dir_plots='/Users/Jaliss/Dropbox/UCSC/CMPE_264/CE264Proj_1/Deliverable/Images'
Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/CE264Proj_1/'#Change this directory to the directory on your computer
Dir2=os.path.join(Dir,'Photos/P1_2/Jpeg')
Array_txt_file="Proj1_2ArrayJ.txt"
fname=os.path.join(Dir2,Array_txt_file)
M_Img_ArrayJ=np.genfromtxt(fname, delimiter=',', skip_header=0)

def HDRMin(Image1,Image2,Image3,a1,a2,Dir): # Zero out every pixel in the image that was 255
    Image1=Image1.copy()
    Image2=Image2.copy()
    Image3=Image3.copy()
    HDRshape=np.shape(Image1)
    HDR=np.zeros((HDRshape[0],HDRshape[1],HDRshape[2]))
    HDR=np.uint8(HDR)
    for i in range(0,HDRshape[0]-1):
        for j in range(0,HDRshape[1]-1):
            if np.min(Image1[i,j,:]) <= 255/a1:
                HDR[i,j,0]=Image2[i,j,0]
                HDR[i,j,1]=Image2[i,j,1]
                HDR[i,j,2]=Image2[i,j,2]
            elif np.min(Image1[i,j,:]) <= 255/a2:
                HDR[i,j,0]=Image3[i,j,0]
                HDR[i,j,1]=Image3[i,j,1]
                HDR[i,j,2]=Image3[i,j,2]
            else:
                HDR[i,j,0]=Image1[i,j,0]
                HDR[i,j,1]=Image1[i,j,1]
                HDR[i,j,2]=Image1[i,j,2]
    HDR=np.uint8(HDR)
    Dir=os.path.join(Dir,'Photos/P1_2/Jpeg')
    plt.savefig(os.path.join(Dir_plots,'HDR_ofStack_ImagesMin.png'))

    return HDR
def HDRAvg (Image1,Image2,Image3,a,Dir):
    Image1=Image1.copy()
    Image2=Image2.copy()
    Image3=Image3.copy()
    HDRshape=np.shape(Image1)
    HDR=np.zeros((HDRshape[0],HDRshape[1],HDRshape[2]))
    HDR=np.uint8(HDR)
    for i in range(0,HDRshape[0]-1):
            for j in range(0,HDRshape[1]-1):
                if np.min(Image1[i,j,:]) <= 254 and np.min(Image2[i,j,:]) <= 254 and np.min(Image3[i,j,:]) <= 254:
                    HDR[i,j,0]=round((Image1[i,j,0]+Image2[i,j,0]+Image3[i,j,0])/3)
                    HDR[i,j,1]=round((Image1[i,j,1]+Image2[i,j,1]+Image3[i,j,1])/3)
                    HDR[i,j,2]=round((Image1[i,j,2]+Image2[i,j,2]+Image3[i,j,2])/3)
                elif np.min(Image1[i,j,:]) <= 254 and np.min(Image2[i,j,:]) <= 254 and np.min(Image3[i,j,:]) >= 255:
                    HDR[i,j,0]=round((Image1[i,j,0]+Image2[i,j,0])/2)
                    HDR[i,j,1]=round((Image1[i,j,1]+Image2[i,j,1])/2)
                    HDR[i,j,2]=round((Image1[i,j,2]+Image2[i,j,2])/2)
                else:
                    HDR[i,j,0]=Image1[i,j,0]
                    HDR[i,j,1]=Image1[i,j,1]
                    HDR[i,j,2]=Image1[i,j,2]
    HDR=np.uint8(HDR)
    Dir=os.path.join(Dir,'Photos/P1_2/Jpeg')
    plt.savefig(os.path.join(Dir_plots,'HDR_ofStack_ImagesAvg_'+str(a)+'.png'))
    #Write to txt file
    '''
    Array_txt_file="Proj1_2ArrayHDR.txt"
    file = open(os.path.join(Dir,Array_txt_file), "w")
    file.close()
    np.savetxt(os.path.join(Dir,Array_txt_file), HDR[:,:,0],fmt='%20.30s', delimiter=',')
    '''
    return HDR
def HDRWAvg (Image1,Image2,Image3,a1,a2,Dir,M_Img_ArrayJ):
    Image1=Image1.copy()
    Image2=Image2.copy()
    Image3=Image3.copy()
    HDRshape=np.shape(Image1)
    HDR1=np.zeros((HDRshape[0],HDRshape[1],HDRshape[2]))
    HDR1=np.uint8(HDR1)
    E1=float(M_Img_ArrayJ[1,6])
    E1T=E1*float(M_Img_ArrayJ[1,1])
    E2=float(M_Img_ArrayJ[11,6])
    E2T=E2*float(M_Img_ArrayJ[11,1])/pow(a1,2)
    E3=float(M_Img_ArrayJ[14,6])
    E3T=E3*float(M_Img_ArrayJ[14,1])/pow(a2,2)
    CW1=E1T/(E1T+E2T+E3T)
    W1=1-CW1 # You want to take the complement to reduce the effects of noisy data
    CW2=E2T/(E1T+E2T+E3T)
    W2=1-CW2 # You want to take the complement to reduce the effects of noisy data
    CW3=E3T/(E1T+E2T+E3T)
    W3=1-CW3 # You want to take the complement to reduce the effects of noisy data
    for i in range(0,HDRshape[0]-1):
        for j in range(0,HDRshape[1]-1):
            if np.min(Image1[i,j,:]) <= 254 and np.min(Image2[i,j,:]) <= 254 and np.min(Image3[i,j,:]) <= 254:
                HDR1[i,j,0]=round((W1*Image1[i,j,0]+W2*Image2[i,j,0]+W3*Image3[i,j,0])/3)
                HDR1[i,j,1]=round((W1*Image1[i,j,1]+W2*Image2[i,j,1]+W3*Image3[i,j,1])/3)
                HDR1[i,j,2]=round((W1*Image1[i,j,2]+W2*Image2[i,j,2]+W3*Image3[i,j,2])/3)
            elif np.min(Image1[i,j,:]) <= 254 and np.min(Image2[i,j,:]) <= 254 and np.min(Image3[i,j,:]) >= 255:
                HDR1[i,j,0]=round((W1*Image1[i,j,0]+W2*Image2[i,j,0])/2)
                HDR1[i,j,1]=round((W1*Image1[i,j,1]+W2*Image2[i,j,1])/2)
                HDR1[i,j,2]=round((W1*Image1[i,j,2]+W2*Image2[i,j,2])/2)
            else:
                HDR1[i,j,0]=Image1[i,j,0]
                HDR1[i,j,1]=Image1[i,j,1]
                HDR1[i,j,2]=Image1[i,j,2]
    HDR1=np.uint8(HDR1)
    Dir=os.path.join(Dir,'Photos/P1_2/Jpeg')
    plt.savefig(os.path.join(Dir_plots,'HDR_ofStack_ImagesWAvg.png'))
    return HDR1
'''
def Linearize(image,Rg,Gg,Bg): #Actual Linearization Function
    image=image.copy()
    Rp=pow(image[:,:,0],Rg)
    Gp=pow(image[:,:,1],Gg)
    Bp=pow(image[:,:,2],Bg)
    Rlin=255*Rp/np.max(Rp)
    Glin=255*Gp/np.max(Gp)
    Blin=255*Bp/np.max(Bp)
    image=np.dstack((Rlin,Glin,Blin))
    image=np.uint8(image) # Must convert every image to uint8 after linearization
    return image

def Image_Lin_Analy(img): #Linearize the image and plot for analysis
    path=os.path.join(Dir2,img)
    image1=cv2.imread(path,1)
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    print "Original image"
    print image1.dtype
    #Linearize each value in the image
    ImageLin=Linearize(image1,1.374,1.399,1.3885)
    print "Linearized Image"
    print ImageLin.dtype
    print np.mean(ImageLin[:,:,0])
    print np.mean(ImageLin[:,:,1])
    print np.mean(ImageLin[:,:,2])
    return ImageLin
'''
def crop(img):
    Shape=np.shape(img)
    img=img[int(Shape[0]/2)-200:int(Shape[0]/2)+500,0:500,:]
    return img

img='IMG_6390.jpeg'
ImageLin1=Image_Lin_Analy(img)
ImageLin1=crop(ImageLin1)
img='IMG_6400.jpeg'
ImageLin2=Image_Lin_Analy(img)
ImageLin2=crop(ImageLin2)
img='IMG_6403.jpeg'
ImageLin3=Image_Lin_Analy(img)
ImageLin3=crop(ImageLin3)

plt.figure()
plt.subplot(131)
plt.imshow(ImageLin1)
plt.show()
plt.title("Original Linearized Image1")

plt.subplot(132)
plt.imshow(ImageLin2)
plt.show()
plt.title("Original Linearized Image2")

plt.subplot(133)
plt.imshow(ImageLin2)
plt.show()
plt.title("Original Linearized Image3")

ImageLin2=ImageLin2.copy()
ImageLin2=np.round(ImageLin2/5)
ImageLin3=ImageLin3.copy()
ImageLin3=np.round(ImageLin3/10)

#Write Plot HDR Photos
print("HDR Simple Start")
ImageHDR3S=HDRMin(ImageLin1,ImageLin2,ImageLin3,5,10, Dir)
plt.figure()
plt.subplot(131)
plt.imshow(ImageHDR3S)
plt.show()
plt.title("Image After HDR Simple with 3 Photos")
print("HDR Simple End")

print("HDR Average Start")
ImageHDR3A=HDRAvg(ImageLin1,ImageLin2, ImageLin3,5,Dir)
plt.subplot(132)
plt.imshow(ImageHDR3A)
plt.show()
plt.title("Image After HDR Avg with 3 Photos")
print("HDR Average End")

print("HDR Weighted Average Start")
ImageHDR3WA=HDRWAvg(ImageLin1,ImageLin2, ImageLin3,5,10,Dir,M_Img_ArrayJ)
plt.subplot(133)
plt.imshow(ImageHDR3WA)
plt.show()
plt.title("Image After HDR Weighted Avg with 3 Photos")
print("HDR Weighted Average End")

end = time.time()
print(str(end - start)+" Seconds")
