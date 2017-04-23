# ProjImgLib: Aids all Proj_1 functions
# ProjImgLib Functionality: Grabs metadata from the images, crops ands outputs brightness values

def drawElines(img1, img2, lines, pts1, pts2):
    import cv2
    import numpy as np
    # img1 - image on which we draw the epilines for the points in img2
    #  lines - corresponding epilines
    r, c, d = np.shape(img1)
    # img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 10)
        img1 = cv2.circle(img1, tuple(pt1), 10, color, 0)
        img2 = cv2.circle(img2, tuple(pt2), 10, color, 0)
    return img1, img2


def PlotMp(matchmat, Keymat, type):
    # If type = 1 then its train image, if its = 3 then its the other image
    import numpy as np
    from matplotlib import patches as pat
    x,y=np.shape(matchmat)
    mp_array = np.zeros((x, 2))  # Match Point Array
    i = 0
    for mp in matchmat[:, type]:
        mp_array[i, 0] = Keymat[int(mp), 0]
        mp_array[i, 1] = Keymat[int(mp), 1]
        i = i + 1
    return mp_array


def KeyPoints(PhotoDir, imgname, Kp):
    import os
    import numpy as np
    kpsize = np.size(Kp)
    print(kpsize)
    keymat = np.zeros((kpsize, 3))
    i = 0
    for keyPoint in Kp:
        keymat[i][0] = keyPoint.pt[0]
        keymat[i][1] = keyPoint.pt[1]
        keymat[i][2] = keyPoint.size
        i = i + 1
    np.savetxt(os.path.join(PhotoDir, imgname.replace('_Undis&Cal_', '_KeyMat_') + '.txt'), keymat, delimiter=',')
    return keymat


def Descriptor(PhotoDir, imgname, des):
    import os
    import numpy as np
    np.savetxt(os.path.join(PhotoDir, imgname.replace('_Undis&Cal_', '_DesMat_') + '.txt'), des, delimiter=',')


def SIFT_Det(Dir, imgfile):
    import cv2
    import os
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray,None)
    img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(os.path.join(Dir, imgfile.replace('.JPG', '') + 's_kp.JPG'), img)
    return kp, des, img


def Surf_Det(Dir, img, gray, imgfile, thresh):
    import cv2
    import os
    surf = cv2.xfeatures2d.SURF_create(thresh)
    # surf.setHessianThreshold(50000)
    print(surf.getHessianThreshold())
    surf.setUpright(True)
    kp, des = surf.detectAndCompute(gray, None)
    img2 = cv2.drawKeypoints(gray, kp, None, (255, 0, 0), 4)
    # img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imgfile = imgfile.replace('.JPG', '')
    print(imgfile)
    cv2.imwrite(os.path.join(Dir, imgfile.replace('_Undis&Cal_', '_s_kp_') + '.JPG'), img2)
    return kp, des, img2


def undistortimage(mtx, img, dist):
    import cv2
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(img, mtx , dist, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]
    return (newcameramtx, roi, dst)


class PhotoData:
    def get_raw_exif(fn):
        import exifread
        import os
        from fractions import Fraction
        # Open image file for reading (binary mode)
        f = open(fn, 'rb')
        # Return Exif tags
        ret = exifread.process_file(f)
        fn = os.path.split(fn)
        fn = fn[1]
        Name = fn
        T = str(float(Fraction(str((ret['EXIF ExposureTime'])))))
        G = str(ret['EXIF ISOSpeedRatings'])
        W = str(ret['EXIF ExifImageWidth'])
        H = str(ret['EXIF ExifImageLength'])
        B = '0'
        G = '0'
        R = '0'
        PicInfo = [Name, T, G, W, H, B, G, R]
        return PicInfo

    def get_jpeg_exif(fn):
        from PIL import Image
        from PIL.ExifTags import TAGS
        import os
        ret = {}
        i = Image.open(fn)
        info = i._getexif()
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            ret[decoded] = value
        fn = os.path.split(fn)
        fn = fn[1]
        Name = fn
        T = ret['ExposureTime']
        T = str(float(T[0]) / float(T[1]))
        G = str(float(ret['ISOSpeedRatings']))
        W = str(ret['ExifImageWidth'])
        H = str(ret['ExifImageHeight'])
        B = '0'
        G = '0'
        R = '0'
        PicInfo = [Name, T, G, W, H, B, G, R]
        return PicInfo

    def GenPhotoMeta(Dir, FileName):
        import os
        import cv2
        import numpy as np
        filelist = os.listdir(Dir)
        Array_txt_file = FileName
        file = open(os.path.join(Dir, Array_txt_file), "w")
        file.close()
        filelist = list(filter(lambda k: '.JPG' in k, filelist))
        filelist = list(filter(lambda k: '.JPG' in k, filelist))
        M_Img_ArrayJ = ['0', '0', '0', '0', '0', '0', '0', '0']
        for img in filelist:
            path = os.path.join(Dir, img)
            PicInfoJ = PhotoData.get_jpeg_exif(path)  # Do not change the name of the jpeg_exif attribute
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
            # crop_image=image[int(PicInfoJ[3])/2-5:int(PicInfoJ[3])/2+5,int(PicInfoJ[4])/2-5:int(PicInfoJ[4])/2+5]
            crop_image = image
            PicInfoJ[5] = str(np.mean(crop_image[:, :, 0]))
            PicInfoJ[6] = str(np.mean(crop_image[:, :, 1]))
            PicInfoJ[7] = str(np.mean(crop_image[:, :, 2]))
            M_Img_ArrayJ = np.vstack((M_Img_ArrayJ, PicInfoJ))
            np.savetxt(os.path.join(Dir, FileName), M_Img_ArrayJ, fmt='%20.30s', delimiter=',')
            PicInfoJtxt = PicInfoJ[0] + ',' + PicInfoJ[1] + ',' + PicInfoJ[2] + ',' + PicInfoJ[3] + ',' + PicInfoJ[
                4] + ',' + \
                          PicInfoJ[5] + ',' + PicInfoJ[6] + ',' + PicInfoJ[7]
            print(PicInfoJtxt)
