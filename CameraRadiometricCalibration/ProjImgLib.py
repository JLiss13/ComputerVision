#ProjImgLib: Aids all Proj_1 functions
#ProjImgLib Functionality: Grabs metadata from the images, crops ands outputs brightness values
def get_raw_exif(fn):
    import exifread
    import os
    from fractions import Fraction
    # Open image file for reading (binary mode)
    f = open(fn, 'rb')
    # Return Exif tags
    ret = exifread.process_file(f)
    fn=os.path.split(fn)
    fn=fn[1]
    Name=fn
    T=str(float(Fraction(str((ret['EXIF ExposureTime'])))))
    G=str(ret['EXIF ISOSpeedRatings'])
    W=str(ret['EXIF ExifImageWidth'])
    H=str(ret['EXIF ExifImageLength'])
    B='0'
    G='0'
    R='0'
    PicInfo=[Name,T,G,W,H,B,G,R]
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
    fn=os.path.split(fn)
    fn=fn[1]
    Name=fn
    T=ret['ExposureTime']
    T=str(float(T[0])/float(T[1]))
    G=str(float(ret['ISOSpeedRatings']))
    W=str(ret['ExifImageWidth'])
    H=str(ret['ExifImageHeight'])
    B='0'
    G='0'
    R='0'
    PicInfo=[Name,T,G,W,H,B,G,R]
    return PicInfo 
    
def PrintToCSVReport(ReportFile,Data1):
    all_content=[]
    with open(ReportFile) as f:
        all_lines = [line.strip() for line in f]
    all_content=all_content+all_lines #aka all_content = all_content+ all_lines
    # Write gathered information to the destination csv file
    all_content=all_content+Data1
    with open(ReportFile, 'w') as f: # 'w'=Write permissions to file
        for line in all_content:
            print >> f, line # Inserts line of appended array to each new line of