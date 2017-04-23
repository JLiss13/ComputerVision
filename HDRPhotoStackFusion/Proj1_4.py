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
Dir3=os.path.join(Dir,'Photos/P1_3/Jpeg')
Dir4=os.path.join(Dir,'Photos/P1_4/Jpeg')

# Loading exposure images from a list
filelist= os.listdir(Dir3)
filelist=filter(lambda k: '.jpeg' in k, filelist)
img_list = [cv2.imread(fn) for fn in filelist]
exposure_times = np.array([0.5,5.0,10.0], dtype=np.float32)

# Tonemap HDR image
tonemap1 = cv2.createTonemapDurand(gamma=2.2)
image1 = tonemap1.process(img_list[0].copy())
tonemap2 = cv2.createTonemapDurand(gamma=1.3)
image2= tonemap2.process(img_list[0].copy())


# Convert datatype to 8-bit and save
image1_8bit = np.clip(image1*255, 0, 255).astype('uint8')
image2_8bit = np.clip(image2*255, 0, 255).astype('uint8')
cv2.imwrite(os.path.join(Dir4,filelist[0]), image1_8bit)
cv2.imwrite(os.path.join(Dir4,filelist[1]), image2_8bit)

end = time.time()
print(str(end - start)+" Seconds")
