import cv2
import os
import numpy as np
import ProjImgLib as PL
from matplotlib import pyplot as plt
from matplotlib import patches as pat
Dir = os.getcwd()
print('Working Directory is: ' + Dir)
PhotoDir = os.path.split(Dir)
PhotoDir = os.path.split(PhotoDir[0])
PhotoDir = os.path.join(PhotoDir[0], 'Photos')
print('Photo directory is: ' + PhotoDir)

files=os.listdir(PhotoDir)
files=list(filter(lambda k: 'Scene_Undis&Cal_' in k, files))
files=list(filter(lambda k: '.JPG' in k, files))

#Obtain the Image
img1=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1a.JPG'))
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1b.JPG'))
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img3=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1c.JPG'))
img3=cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
Kp1, des1 = orb.detectAndCompute(img1, None)
Kp2, des2 = orb.detectAndCompute(img2, None)



#Save the Keypoints from each image to a .txt file
Keymat1=PL.KeyPoints(PhotoDir,'Scene_Undis&Cal_1a',Kp1)
Keymat2=PL.KeyPoints(PhotoDir,'Scene_Undis&Cal_1c',Kp2)
#Keymat3=PL.KeyPoints(PhotoDir,'Scene_Undis&Cal_1c',Kp3)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(Kp2[m.trainIdx].pt)
        pts1.append(Kp1[m.queryIdx].pt)

pts1=np.int32(pts1)
pts2=np.int32(pts2)
# Find F & E for images
mtx=[[4.445119981167400510e+03,0.000000000000000000e+00,2.517778628297769501e+03],
[0.000000000000000000e+00,4.425795137571955820e+03,1.771483728014967710e+03],
[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]]

# F,mask=cv2.findFundamentalMat(mp_array1,mp_array2, 2, 3)
F,mask=cv2.findFundamentalMat(pts1,pts1, cv2.FM_LMEDS)
print("F \n")
print(F)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = PL.drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = PL.drawlines(img2,img1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()