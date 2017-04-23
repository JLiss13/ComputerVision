import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

import time
start = time.time()
#Maintenance
plt.close('all')
Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/Baudin_Liss_CE264Proj_1/'
Dir_plots=os.path.join(Dir,'Deliverable/Images') # Directory to Deliverable Images
 #Directory to Radiometric Calibration
Dir2=os.path.join(Dir,'Photos/P1_2/Jpeg')
Dir3=os.path.join(Dir,'Photos/P1_3/Jpeg')
Array_txt_file="Proj1_2ArrayJ.txt"
fname=os.path.join(Dir2,Array_txt_file)
M_Img_ArrayJ=np.genfromtxt(fname, delimiter=',', skip_header=1)

# Functions
def Image_Lin_Analy(img): #Linearize the image and plot for analysis
    path=os.path.join(Dir2,img)
    image1=cv2.imread(path,1)
    print "Original image"
    print image1.dtype
    #Linearize each value in the image
    ImageLin=Linearize(image1,1.374,1.399,1.3885)
    print "Linearized Image"
    print ImageLin.dtype
    print np.mean(ImageLin[:,:,0])
    print np.mean(ImageLin[:,:,1])
    print np.mean(ImageLin[:,:,2])
    #plot histograms for each image
    plt.figure()
    color = ('r','g','b')
    for i,col in enumerate(color):
        histr = cv2.calcHist([ImageLin],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title(img)
    plt.show()
    plt.savefig(os.path.join(Dir_plots,'HistogramLinearized_'+str(img)))
    return ImageLin

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

'''
# Plotting Brightness vs. Exposure for Raw Files
Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/CE264Proj_1/Photos/P1_1/Raw'
Array_txt_file="Proj1_1ArrayR.txt"
fname=os.path.join(Dir,Array_txt_file)
RawArray=np.genfromtxt(fname, delimiter=',', skip_header=1)

plt.figure()
plt.plot(RawArray[:,1],RawArray[:,7]) #Red
plt.ylabel('B_p^g')
plt.xlabel('Exposure Time (seconds)')
plt.show()
'''
# Plotting Linearization of Jpeg Files

img='IMG_6390.jpeg'
ImageLin1=Image_Lin_Analy(img)
cv2.imwrite(os.path.join(Dir3,'Linearization_of_'+str(img)),ImageLin1)
ImageLin1 = cv2.cvtColor(ImageLin1,cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(ImageLin1)
plt.title('Linearized ' +str(img))
plt.show()

img='IMG_6400.jpeg'
ImageLin2=Image_Lin_Analy(img)
cv2.imwrite(os.path.join(Dir3,'Linearization_of_'+str(img)),ImageLin2)
ImageLin2 = cv2.cvtColor(ImageLin2,cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(ImageLin2)
plt.title('Linearized ' +str(img))
plt.show()


img='IMG_6403.jpeg'
ImageLin3=Image_Lin_Analy(img)
cv2.imwrite(os.path.join(Dir3,'Linearization_of_'+str(img)),ImageLin3)
ImageLin3 = cv2.cvtColor(ImageLin3,cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(ImageLin3)
plt.title('Linearized ' +str(img))
plt.show()

end = time.time()
print(str(end - start)+" Seconds")
