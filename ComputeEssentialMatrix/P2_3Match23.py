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
img1=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1b.JPG'))
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1c.JPG'))
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
Kp1, des1 = orb.detectAndCompute(img1, None)
Kp2, des2 = orb.detectAndCompute(img2, None)

#Save the Keypoints from each image to a .txt file
Keymat1=PL.KeyPoints(PhotoDir,'Scene_Undis&Cal_1b',Kp1)
Keymat2=PL.KeyPoints(PhotoDir,'Scene_Undis&Cal_1c',Kp2)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches12 = bf.match(des1,des2)
size_matches12=np.size(matches12)
# Sort them in the order of their distance.
matches12 = sorted(matches12, key = lambda x:x.distance)
# matches12=matches12[0::int(size_matches12/100)]
matchsize = np.size(matches12)
matchmat12 = np.zeros((matchsize, 4))
i = 0
for match in matches12:
    matchmat12[i][0] = match.distance
    matchmat12[i][1] = match.trainIdx
    matchmat12[i][2] = match.imgIdx
    matchmat12[i][3] = match.queryIdx
    i = i + 1
#http://docs.opencv.org/trunk/d1/d89/tutorial_py_orb.html is the tutorial for ORB
#http://docs.opencv.org/3.0-beta/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html documention for matching
#Match between scene 1a and scene 1b

# matchesMask = [[0,0] for i in range(len(matches))]
# draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)
MatchedImg = cv2.drawMatches(img1, Kp1, img2, Kp2, matches12, None, flags=2)
#MatchedImg = cv2.drawMatches(img1, Kp1, img2, Kp2, matches, None,**draw_params)
fig, ax = plt.subplots(1)
ax.imshow(MatchedImg)

#Image 1
mp_array1 = PL.PlotMp(matchmat12,Keymat1,3) #Look to the ProjImgLib queryidx is for the left image
np.savetxt(os.path.join(PhotoDir, 'mp_array1_points.txt'), mp_array1, delimiter=',')
x,y=np.shape(mp_array1)
#Print blue dots on image

#http://stackoverflow.com/questions/34902477/drawing-circles-on-image-with-matplotlib-and-numpy
for i in range(x) :
    Circle1 = pat.Circle((mp_array1[i,0],mp_array1[i,1]), 50)
    ax.add_patch(Circle1)
# plt.show()

#Image 2
mp_array2 = PL.PlotMp(matchmat12,Keymat2,1) #Look to the ProjImgLib trainidx is for the right image
np.savetxt(os.path.join(PhotoDir, 'mp_array3_points.txt'), mp_array2, delimiter=',')

x,y=np.shape(mp_array2)
#Print blue dots on image

for i in range(x) :
    Circle2 = pat.Circle((5184+mp_array2[i,0],mp_array2[i,1]), 50)
    ax.add_patch(Circle2)
# plt.show()

plt.title('Points from Scene 1b and Scene 1c')
mtx=[[4.445119981167400510e+03,0.000000000000000000e+00,2.517778628297769501e+03],
[0.000000000000000000e+00,4.425795137571955820e+03,1.771483728014967710e+03],
[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]]
mtx=np.array(mtx)

E_ideal,mask_e=cv2.findEssentialMat(mp_array1,mp_array2,4445,(2517,1771))
print("OpenCV E \n")
print(E_ideal)

#http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=findfundamentalmat#cv2.findFundamentalMat
F,mask_f=cv2.findFundamentalMat(mp_array1,mp_array2,cv2.FM_RANSAC, 1, .99) #4th param is pixel disparity threshold & 5th is confidence percent
print("F \n")
print(F)

E_calc=np.matmul(np.transpose(mtx), np.matmul(F,mtx))
print("Calculated E \n")
print(E_calc)
size_outliers=np.size(mask_f[mask_f<1])
size_inliers=np.size(mask_f[mask_f>0])
print("Number of inliers: "+ str(size_inliers)+"\n")
print("Number of outliers: "+ str(size_outliers)+"\n")

# Upload new set of image arrays
img1=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1b.JPG'))
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1c.JPG'))
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

#Draw Epipolar lines for Image 1 to 2, Image 2 to 3, Image 1 to 3
pts1=[]
pts2=[]

#Create inlier and outlier points for plotting
pts1 = mp_array1[mask_f.ravel()==1]
pts2 = mp_array2[mask_f.ravel()==1]
pts1_out = mp_array1[mask_f.ravel()==0]
x_pts1_out,y_pts1_out=np.shape(pts1_out)
pts1_out=pts1_out[int(x_pts1_out/2):x_pts1_out]
pts2_out = mp_array2[mask_f.ravel()==0]
x_pts2_out,y_pts2_out=np.shape(pts2_out)
pts2_out=pts2_out[int(x_pts2_out/2):x_pts2_out]
pts1=np.int32(pts1)
pts2=np.int32(pts2)
np.savetxt(os.path.join(PhotoDir,'Match23_pts1.txt'),pts1,delimiter=',')
np.savetxt(os.path.join(PhotoDir,'Match23_pts2.txt'),pts2,delimiter=',')
pts1_out=np.int32(pts1_out)
pts2_out=np.int32(pts2_out)

# Find epilines corresponding to points in right image (second image) and
# Drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = PL.drawElines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img7,img8 = PL.drawElines(img2,img1,lines2,pts2,pts1)

#Image 1 & 2
#Image 1 with Epi
x,y=np.shape(pts1)
plt.figure()

#Plot Inliers
for i in range(x) :
    cv2.circle(img5, (pts1[i,0], pts1[i,1]), 20, (0, 255, 0), thickness=20, lineType=8,
               shift=0)
    # Circle1 = pat.Circle((pts1[i,0],pts1[i,1]), 50, facecolor="green")
    # ax2.add_patch(Circle1)
#Plot Outliers
x_out,y_out=np.shape(pts1_out)
for i in range(x_out) :
    cv2.circle(img5, (pts1_out[i, 0], pts1_out[i, 1]), 5, (255, 0, 0), thickness=10, lineType=8,
               shift=0)
    # Circle1out = pat.Circle((pts1_out[i,0],pts1_out[i,1]), 30, facecolor="red")
    # ax2.add_patch(Circle1out)
plt.imshow(img5)
plt.title("Image 1b with Inliers, Outliers and Epipolar Lines")
img5=cv2.cvtColor(img5,cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(PhotoDir,"Scene_1b1c_Epi.JPG"),img5)
#Image 2 with Epi
x,y=np.shape(pts2)
plt.figure()

#Plot Inliers
for i in range(x) :
    cv2.circle(img7, (pts2[i,0], pts2[i,1]), 20, (0, 255, 0), thickness=20, lineType=8,
               shift=0)
    # Circle2out = pat.Circle((pts2[i,0],pts2[i,1]), 50,facecolor="green")
    # ax3.add_patch(Circle2out)

#Plot Outliers
x_out,y_out=np.shape(pts2_out)
for i in range(x_out-1) :
    cv2.circle(img7, (pts2_out[i,0], pts2_out[i,1]), 5, (255, 0, 0), thickness=10, lineType=8,
               shift=0)
    # Circle1out = pat.Circle((pts2_out[i,0],pts2_out[i,1]), 30, facecolor="red")
    # ax3.add_patch(Circle1out)
plt.imshow(img7)
plt.title("Image 1c with Inliers, Outliers and Epipolar Lines")
img7=cv2.cvtColor(img7,cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(PhotoDir,"Scene_1c1b_Epi.JPG"),img7)