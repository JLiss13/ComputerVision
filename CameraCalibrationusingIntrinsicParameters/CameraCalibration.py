import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import ProjImgLib as PL
#Makes gray photos of the checkboards
plt.close('all')
Dir = os.getcwd()
print('Working Directory is: ' + Dir)
PhotoDir = os.path.split(Dir)
PhotoDir = os.path.split(PhotoDir[0])
PhotoDir = os.path.join(PhotoDir[0], 'Photos')
#PhotoDir = os.path.join(PhotoDir[0], 'Photos/Camera2')
print('Photo directory is: ' + PhotoDir)
M_Img_ArrayJ = np.genfromtxt(os.path.join(PhotoDir, 'P2_PhototxtFile'), delimiter=',', skip_header=1, dtype=(None))

# Camera calibration
# termination criteria
x = 7
y = 7
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x * y, 3), np.float32)
objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = os.listdir(PhotoDir)
images = list(filter(lambda k: 'IMG_3' in k, images))
#images = list(filter(lambda k: 'IMG_6' in k, images))
images = list(filter(lambda k: '.JPG' in k, images))
for fname in images:
    print(fname)
    img = cv2.imread(os.path.join(PhotoDir, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (x, y), None)
    print(ret)  # If it returns false then the image does not have a chessboard in it
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (x, y), corners, ret)
        plt.figure()
        plt.imshow(img)
        plt.title('Chessboard ' + fname)
        plt.show()
        cv2.imwrite(os.path.join(PhotoDir, 'gray_' + fname.replace('.JPG','') + '.png'), gray)

# Using the findChessBoard Outputs we can determine the calibration
[ret, mtx, dist, rvecs, tvecs] = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savetxt(os.path.join(PhotoDir,'CameraMat.txt'),mtx,delimiter=',')
np.savetxt(os.path.join(PhotoDir,'Cameradist.txt'),dist,delimiter=',')
np.savetxt(os.path.join(PhotoDir,'Camerarvecs.txt'),rvecs,delimiter=',') # Rotation vectors
np.savetxt(os.path.join(PhotoDir,'Cameratvecs.txt'),tvecs,delimiter=',') # Translation vectors

#Determine the camera matrix calibration and distortion coefficients
for imgname in images:
    img = cv2.imread(os.path.join(PhotoDir, imgname))
    print(imgname)
    newcameramtx, roi, dst = PL.undistortimage(mtx, img, dist)
    cv2.imwrite(os.path.join(PhotoDir,imgname.replace('.JPG','') + 'Cal.png'), dst)

# Projection error
tot_error=0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error = tot_error+error
    print(error)
print("total error: ", tot_error/len(objpoints))