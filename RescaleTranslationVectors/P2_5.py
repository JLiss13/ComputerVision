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

#Scene 1a to 1b, 1a to 1c and 1b to 1c

r_L_12= [0.622158 ,   0.02867408 , 0.78236642]
R_R_L_12=[[-0.49027732,  0.00365481,  0.87155883],
 [ 0.03961878, -0.99886406,  0.02647537],
 [ 0.87066555,  0.04751037,  0.48957559]]
r_L_13=[-0.1736334,   0.08105081, -0.98146941]
R_R_L_13=[[ 0.93329118,  0.25460524, -0.25326614],
 [ 0.22769887, -0.96489064, -0.13091709],
 [-0.27770631,  0.06451535, -0.95849725]]
r_L_32=[ 0.16048395,  0.07643708,  0.98407432]
R_R_L_23=[[ 0.96849065, -0.04526156, -0.24490255],
 [ 0.04185303,  0.99894106, -0.01910704],
 [ 0.24550803,  0.00825507,  0.96935941]]
r_L_31_1=-np.dot(np.transpose(R_R_L_13),r_L_13)
r_L_23_1=-np.dot(np.transpose(R_R_L_12),r_L_32)
# r_L_21_1=-np.dot(np.transpose(R_R_L_12),r_L_21)
r_L_21_1=-np.array(r_L_12)
A=[[2*np.dot(np.transpose(r_L_23_1),r_L_23_1), 2*np.dot(r_L_23_1,r_L_31_1)],
   [2*np.dot(r_L_23_1,r_L_31_1) , 2*np.dot(np.transpose(r_L_31_1),r_L_31_1)]]
b=[2*np.dot(r_L_21_1,r_L_23_1),2*np.dot(r_L_21_1,r_L_31_1)]
x=np.matmul(np.linalg.inv(A),np.reshape(b,(2,1)))
print(x)

#Check the
r_L_12_1=np.array(r_L_21_1)
total_wo=np.dot(r_L_12_1,r_L_12_1)+2*np.dot(r_L_23_1,r_L_12_1)+2*np.dot(r_L_12_1,r_L_31_1)+\
         np.dot(r_L_23_1,r_L_23_1)+2*np.dot(r_L_23_1,r_L_31_1)+np.dot(r_L_31_1,r_L_31_1)
print("Check without beta and gamma \n")
print(total_wo)
total=np.dot(r_L_12_1,r_L_12_1)+2*x[0]*np.dot(r_L_23_1,r_L_12_1)+2*x[1]*np.dot(r_L_12_1,r_L_31_1)+\
      pow(x[0],2)*np.dot(r_L_23_1,r_L_23_1)+2*x[0]*x[1]*np.dot(r_L_23_1,r_L_31_1)+pow(x[1],2)*np.dot(r_L_31_1,r_L_31_1)
print("Check with beta and gamma \n")
print(total)
