_author_='Jordan'
import ProjImgLib as PL
import os
Dir=os.getcwd()
print('Working Directory is: ' + Dir)
PhotoDir=os.path.split(Dir)
PhotoDir=os.path.split(PhotoDir[0])
PhotoDir = os.path.join(PhotoDir[0], 'Photos')
#PhotoDir = os.path.join(PhotoDir[0], 'Photos/Camera2')
print('Photo directory is: ' + PhotoDir)
PL.PhotoData.GenPhotoMeta(PhotoDir,'P2_PhototxtFile')
print('Metadata about each photo is saved as P2_PhototxtFile in directory: '+ PhotoDir)