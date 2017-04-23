import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
#Maintenance

Dir = os.getcwd()
print('Working Directory is: ' + Dir)
PhotoDir = os.path.split(Dir)
PhotoDir = os.path.split(PhotoDir[0])
PhotoDir = os.path.join(PhotoDir[0], 'Photos')
print('Photo directory is: ' + PhotoDir)
img1=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1a.JPG'))
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1b.JPG'))
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
R_L_12=np.matrix([[-0.49027732 , 0.00365481 , 0.87155883],
 [ 0.03961878, -0.99886406,  0.02647537],
 [ 0.87066555 , 0.04751037,  0.48957559]])
K=np.matrix([[4.445119981167400510e+03,0.000000000000000000e+00,2.517778628297769501e+03],
[0.000000000000000000e+00,4.425795137571955820e+03,1.771483728014967710e+03],
[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])
img3=cv2.warpPerspective(img1,K*R_L_12*K.I,(3000,2000))
plt.imshow(img3)