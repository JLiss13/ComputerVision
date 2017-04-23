import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
#Maintenance

Dir = os.getcwd()
print('Working Directory is: ' + Dir)
PhotoDir = os.path.split(Dir)
PhotoDir = os.path.split(PhotoDir[0])
PhotoDir = os.path.join(PhotoDir[0], 'Photos')
print('Photo directory is: ' + PhotoDir)

# Input Essential Matricies

E_13=[[-0.16218189 ,  0.67880286 , 0.08474815],
 [ 0.67714071 , 0.17409751, -0.10541694],
 [ 0.0101591,  -0.07504744, -0.00799476]]

print("E_13 \n")
print(E_13)
print("\n")

# Determine the r_L. Take the SVD of E to determine the r_L. Also force E to rank=2

U_13,S_13,V_T_13=np.linalg.svd(E_13)
S_13[2]=0
S_13=np.diag(S_13)
E_13=np.matmul(np.matmul(U_13,S_13),V_T_13)
U_13,S_13,V_T_13=np.linalg.svd(E_13)
r_L_13=V_T_13[2,:]
print("r_L_13 \n")
print(r_L_13)

#Determine the Rotation Matrix
W=[[0,-1,0],[1,0,0],[0,0,1]]
W=np.matrix(W)
# W=W.T
K=[[4.445119981167400510e+03,0.000000000000000000e+00,2.517778628297769501e+03],
[0.000000000000000000e+00,4.425795137571955820e+03,1.771483728014967710e+03],
[0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]]
K=np.matrix(K)
f_l=4445
f_r=f_l

RL_1=np.matmul(np.matmul(U_13,W),V_T_13)
RL_2=np.matmul(np.matmul(U_13,W.T),V_T_13)
RL_1=RL_2
XL_l_list=np.genfromtxt(os.path.join(PhotoDir,"Match13_pts1.txt"),delimiter=",")
XL_l_list=np.hstack((XL_l_list,np.ones((np.shape(XL_l_list)[0],1))))
XL_l_list=np.matrix(XL_l_list)
XR_r_list=np.genfromtxt(os.path.join(PhotoDir,"Match13_pts2.txt"),delimiter=",")
XR_r_list=np.hstack((XR_r_list,np.ones((np.shape(XR_r_list)[0],1))))
Pz_L_mat=np.zeros((np.shape(np.array(XL_l_list))[0],1))
for i in range(np.shape(XL_l_list)[0]):
    XL_l_c=np.matmul(np.linalg.inv(K),XL_l_list[i,:].T)
    XR_r=XR_r_list[i, :]
    Pz_L=f_l*(np.dot((f_r*RL_1[0,:]-XR_r[0]*RL_1[2,:]),-r_L_13))/(np.dot((f_r*RL_1[0,:]-XR_r[0]*RL_1[2,:]),XL_l_c))
    Pz_L_mat[i]=int(Pz_L)
print("R_R_L\n")
print(RL_1)
print("Pz_L_12\n")
print(Pz_L_mat)
print("Number of Negative values")
print(np.shape(Pz_L_mat[Pz_L_mat < 0]))
# Show in the second image of the pair the original location of the inliers
img1=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1a.JPG'))
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.imread(os.path.join(PhotoDir,'Scene_Undis&Cal_1c.JPG'))
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

plt.figure()
#Determine what E is with the rotation matrix and skew symmetric matrix of vector r_L_12
r_skew=[[0, -r_L_13[2], r_L_13[1]],[r_L_13[2],0,-r_L_13[0]],[-r_L_13[1],r_L_13[0],0]]
E_12=[[  0.20437851, -12.01964867,  0.11444134],
      [ 13.41421607 , -5.15929358 ,-10.6316888 ],
 [ -0.2665049  ,  7.34594954  , 0.04354355]]
E_12=-(np.array(E_12))
#E=-np.matmul(r_skew,RL_1)
#E=-np.array(r_skew)
XR_r_list_new=np.zeros((np.shape(XR_r_list)[0],3))
#Need to convert camera pixel in pixel reference frame to camera reference frame and then perform rotation.
#Then take point and rotate it back to pixel reference frame
#Plot Inliers
error=np.zeros((np.shape(XR_r_list)[0],1))
for i in range(np.shape(XR_r_list)[0]):
    cv2.circle(img2, (int(XR_r_list[i,0]), int(XR_r_list[i,1])), 50, (0, 255, 0), thickness=20, lineType=8, shift=0)
    XL_l_list_C=K.I*np.reshape(XL_l_list[i,:],(3,1)) # Convert from Camera to Pixel
    XL_l_list_temp=np.array(np.matmul(RL_1,XL_l_list_C)-0.29051651*np.reshape(np.dot(RL_1,r_L_13),(3,1)))# Perform rotation + translation
    # XL_l_list_temp=np.array(np.matmul(RL_1,XL_l_list_C)-np.reshape(np.dot(RL_1,r_L_13),(3,1)))# Perform rotation + translation
    XL_l_list_temp=np.matmul(K, XL_l_list_temp)
    XR_r_list_new[i,:]=np.reshape(XL_l_list_temp/XL_l_list_temp[2],(1,3))
    error[i] = np.linalg.norm(XR_r_list_new[i, :]) - np.linalg.norm(XR_r_list[i, :])
    cv2.circle(img2, (int(XR_r_list_new[i, 0]), int(XR_r_list_new[i, 1])), 20, (255, 0, 0), thickness=20, lineType=8, shift=0)
plt.imshow(img2)
print("XR_r_list_new \n")
print(XR_r_list_new)
plt.title("Reprojection of Match Pts from Image 1 onto Image 3")
img2=cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(PhotoDir,"P2_4_Reproject_13.JPG"),img2)
print(np.average(error))