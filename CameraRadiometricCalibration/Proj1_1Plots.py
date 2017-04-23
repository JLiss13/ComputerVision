from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import stats
import cv2
#Maintenance
plt.close('all')


Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/Baudin_Liss_CE264Proj_1/'
Dir2=os.path.join(Dir,'Photos/P1_1/Jpeg')
Dir_plots=os.path.join(Dir,'Deliverable/Images')
def ProjPlots(T,B_p,Color): # Plot histogram of color channels
    print('\n'+Color+' Channel') # Color Channel
    plt.figure()
    plt.plot(T,B_p)
    plt.title('B\' vs T '+str(Color))
    plt.ylabel('B\'')
    plt.xlabel('Exposure Time (seconds)')
    plt.show()
    plt.grid(True)
    plt.savefig(os.path.join(Dir_plots,'Jpeg_'+str(Color)+'_Channel_before.png'))

    L_Bp=np.log(B_p)
    L_T=np.log(T)

    plt.figure()
    plt.plot(L_T,L_Bp)
    plt.title('Log(B\') vs Log(T) ' +str(Color))
    plt.ylabel('log(B\')')
    plt.xlabel('log(Exposure Time (seconds))')

    slope, intercept, r_value, p_value, std_err= stats.linregress(L_T,L_Bp)
    #y=mx+b m=slope, b=intercept, rvalue=correlation coefficient
    # pvalue= two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
    # stderr=Standard error of the estimated gradient.
    fit="y="+str(slope)+"x+"+str(intercept)
    #plt.text(max(L_T)/2,max(L_Bp)/2.0,fit)
    print('B\' vs T')
    print("The mean is at least " + str(np.mean(L_Bp)/np.sqrt(std_err)) +" times larger than the standard derivation")
    print("Linear function is" +fit)
    print("1/g is"+"="+str(slope),"K is"+"="+str(intercept))
    g=1/slope
    print("g="+str(g))

    plt.plot(L_T,slope*L_T+intercept,'r--')
    plt.show()
    plt.grid(True)
    plt.savefig(os.path.join(Dir_plots,'Jpeg_'+str(Color)+'_Channel_log.png'))

    B=pow(B_p,1/slope)
    B_norm=B
    #B_norm=255*B/np.max(B)
    plt.figure()
    plt.plot(T,B_norm)
    plt.title('B vs T '+str(Color))
    plt.ylabel('B=B\'^g ')
    plt.xlabel('Exposure Time (seconds)')
    slope, intercept, r_value, p_value, std_err= stats.linregress(T,B_norm)
    fit="y="+str(slope)+"x+"+str(intercept)
    #plt.text(max(T)/2,max(B)/2,fit)
    print('\n B vs T')
    print("The mean is at least " + str(np.mean(B_norm)/np.sqrt(std_err)) +" times larger than the standard derivation")
    print("Linear function is" +fit)
    #plt.plot(T,slope*T+intercept,'r--') #Keeps giving an error
    plt.show()
    plt.grid(True)
    plt.plot(T,slope*T+intercept,'r--')
    plt.show()
    plt.grid(True)
    plt.savefig(os.path.join(Dir_plots,'Jpeg_'+str(Color)+'_Channel_after.png'))
    return g
def Linearize(image,Rg,Gg,Bg): #Actual Linearization Function
    image=image.copy()
    Rp=pow(image[:,:,0],Rg)
    Gp=pow(image[:,:,1],Gg)
    Bp=pow(image[:,:,2],Bg)
    Rlin=255*Rp/np.max(Rp)
    Glin=255*Gp/np.max(Gp)
    Blin=255*Bp/np.max(Bp)
    image=np.dstack((Rlin,Glin,Blin))
    image=np.uint8(image) # Must convert every image to uint8 after linearization
    return image
'''
# Plotting Brightness vs. Exposure for Raw Files
Dir='/Users/Jaliss/Dropbox/UCSC/CMPE_264/CE264Proj_1/Photos/P1_1/Raw'
Array_txt_file="Proj1_1ArrayR.txt"
fname=os.path.join(Dir,Array_txt_file)
RawArray=np.genfromtxt(fname, delimiter=',', skip_header=1)

plt.figure()
plt.plot(RawArray[:,1],RawArray[:,7]) #Red
plt.ylabel('B_p^g')
plt.xlabel('Exposure Time (seconds)')
plt.show()
'''
# Plotting Brightness vs. Exposure for Jpeg Files

Array_txt_file="Proj1_1ArrayJ.txt"
fname=os.path.join(Dir2,Array_txt_file)
RawArray=np.genfromtxt(fname, delimiter=',', skip_header=1)

# RED CHANNEL
Rg=ProjPlots(RawArray[:,1],RawArray[:,5],'Red')

# GREEN CHANNEL
Gg=ProjPlots(RawArray[:,1],RawArray[:,6],'Green')

# BLUE CHANNEL
Bg=ProjPlots(RawArray[:,1],RawArray[:,7],'Blue')

image1=os.path.join(Dir2,"IMG_6332.JPG")
image1=cv2.imread(image1,1)
plt.figure()
plt.imshow(image1)
plt.show()

imageLin=Linearize(image1,Rg,Gg,Bg)
plt.figure()
plt.imshow(imageLin)
plt.show()
