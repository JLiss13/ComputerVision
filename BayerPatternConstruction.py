#Hw2_1 Computer Vision
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
def aver(i,j,num,raw_image):
    raw_image[i,j,2]=(raw_image[i-1,j,2]+raw_image[i+1,j,2])/2
    raw_image[i,j,0]=(raw_image[i,j-1,0]+raw_image[i,j+1,0])/2
    return
plt.close("all")
Image_URL=raw_input("What is the directory of image?")
image=cv2.imread(Image_URL,1)
image=image.copy()
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image=np.uint8(image)
# Each row has RGB contributions, the produced color is equal contributions from each RGB.
# To implement the BAYER algo we need to interpolate each color from each pixel separately.
# G, R, G; B, G, B;G, R, G
plt.figure()
plt.subplot(221)
plt.imshow(image, interpolation='none')
plt.show()

cropped_image=image[70:75,170:175]#
print cropped_image
plt.subplot(222)
plt.imshow(cropped_image, interpolation='none')
plt.show()

#Perform on all pixels not edges or corners
raw_image=np.array(image, dtype='int16')
size=np.shape(raw_image)
M=size[0]
N=size[1]
for j in range(1,N-1):
    for i in range(1,M-1):
        #GRN Pixel
        if j % 2 == 1 and i % 2 == 1:
            raw_image[i,j,2]=(image[i-1,j,2]+image[i+1,j,2])/2.0
            raw_image[i,j,0]=(image[i,j-1,0]+image[i,j+1,0])/2.0
        elif j % 2 == 0 and i % 2 == 0:
            raw_image[i,j,2]=(image[i,j-1,2]+image[i,j+1,2])/2.0
            raw_image[i,j,0]=(image[i-1,j,0]+image[i+1,j,0])/2.0
        #RED Pixel
        elif j % 2 == 1 and i % 2 == 0:
            raw_image[i,j,1]=(image[i-1,j,1]+image[i,j-1,1]+image[i+1,j,1]+image[i,j+1,1])/4.0
            raw_image[i,j,0]=(image[i-1,j-1,0]+image[i-1,j+1,0]+image[i+1,j-1,0]+image[i+1,j+1,0])/4.0
        #BLU Pixel
        elif j % 2 == 0 and i % 2 == 1:
            raw_image[i,j,1]=(image[i-1,j,1]+image[i,j-1,1]+image[i+1,j,1]+image[i,j+1,1])/4.0
            raw_image[i,j,2]=(image[i-1,j-1,2]+image[i-1,j+1,2]+image[i+1,j-1,2]+image[i+1,j+1,2])/4.0
plt.subplot(223)
plt.imshow(raw_image)
plt.show()

croppedraw_image=raw_image[70:75,170:175] #
print croppedraw_image
plt.subplot(224)
plt.imshow(croppedraw_image, interpolation='none')
plt.show()

Imagedirectory = raw_input("Where do you want the image saved?")
os.chdir(Imagedirectory)
plt.figure()
plt.imshow(raw_image)
plt.show()
plt.savefig('bayer_applied.png')
