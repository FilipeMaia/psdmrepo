#(c) Coded by Alvaro Sanchez-Gonzalez 2014

#Functions related with the XTCAV pulse retrieval

import numpy as np
import scipy.io
import scipy.interpolate
import scipy.stats.mstats 
import matplotlib.pyplot as plt

import scipy.ndimage as im 
import time
import math
import IPython


def ProcessXTCAVImage(image,ROI,n):
#Obtain the statistics (profiles, center of mass, etc) of an xtcav image. 
#Arguments:
#   image: 3d numpy array where the first index always has one dimension (it will become the bunch index), the second index correspond to y, and the third index corresponds to x
#   ROI: region of interest of the image, contain x and y axis
#   n: number of bunches to extract from the initial single image
#Output
#   imageStats: list with the image statistics for each bunch in the image
    
    #Splitting the image in more than one if necessary.
    if n>1:
        image=SplitImage(image,n)
    
    #obtain the number of bunches for the image. In principle this should be equal to n    
    nb=image.shape[0];
        
    #For the image for each bunch we retrieve the statistics and add them to the list    
    imageStats=[];    
    for i in range(0,nb):
        imageStats.append(GetImageStatistics(image[i,:,:],ROI['x'],ROI['y']))
        
    return imageStats
    
def GetCenterOfMass(image,x,y):
#Gets the center of mass of an image 
#Arguments:
#   image: 2d numpy array where the firs index correspond to y, and the second index corresponds to x
#   x,y: vectors of the image
#Output
#   x0,y0 coordinates of the center of mass 
  
    profilex=np.sum(image,0);     
    x0=np.dot(profilex,np.transpose(x))/np.sum(profilex)
    profiley=np.sum(image,1);     
    y0=np.dot(profiley,y)/np.sum(profiley)
    
    return x0,y0
    
    
def GetImageStatistics(image,x,y):
#Obtain all the statistics (profiles, center of mass, etc) of an image
#Arguments:
#   image: 2d numpy array where the first index correspond to y, and the second index corresponds to x
#   x,y: vectors of the image
#Output
#   imageStats: struct that contains the different statistics, including the center of mass 

    imFrac=np.sum(image)    #Total area of the image: Since the original image is normalized, this should be on for on bunch retrievals, and less than one for multiple bunches

    xProfile=np.sum(image,0);                           #Profile projected onto the x axis
    xCOM=np.dot(xProfile,np.transpose(x))/imFrac        #X position of the center of mass
    xRMS= np.sqrt(np.dot((x-xCOM)**2,xProfile)/imFrac); #Standard deviation of the values in x
    ind=np.where(xProfile >= np.amax(xProfile)/2)[0];   
    xFWHM=np.abs(ind[-1]-ind[0]+1);                     #FWHM of the X profile
            
    yProfile=np.sum(image,1);                           #Profile projected onto the y axis
    yCOM=np.dot(yProfile,y)/imFrac                      #Y position of the center of mass
    yRMS= np.sqrt(np.dot((y-yCOM)**2,yProfile)/imFrac); #Standard deviation of the values in y
    ind=np.where(yProfile >= np.amax(yProfile)/2);
    yFWHM=np.abs(ind[-1]-ind[0]);                       #FWHM of the Y profile
    
    yCOMslice=np.dot(np.transpose(image),y)/xProfile;   #Y position of the center of mass for each slice in x
    yCOMslice[np.isnan(yCOMslice)]=yCOM                 #If it is NaN we just set the total y center of mass
    distances=np.outer(np.ones(yCOMslice.shape[0]),y)-np.outer(yCOMslice,np.ones(image.shape[0]))    #For each point of the image, the distance to the y center of mass of the corresponding slice
    yRMSslice= np.sqrt(np.sum(np.transpose(image)*((distances)**2),1)/xProfile)         #Width of the distribution of the points for each slice around the y center of masses                  
    yRMSslice[np.isnan(yRMSslice)]=0;                   #If it is NaN we just set it to 0
    
    if imFrac==0:   #What to to if the image was effectively full of zeros
        xCOM=(x[-1]+x[0])/2.0;
        xRMS=0;
        xFWHM=0;
        yCOM=(y[-1]+y[0])/2.0;
        yRMS=0;
        yFWHM=0;
        yCOMslice[np.isnan(yCOMslice)]=yCOM
    
    imageStats={
        'imFrac':imFrac,
        'xProfile':xProfile,
        'xCOM':xCOM,
        'xRMS':xRMS,
        'xFWHM':xFWHM,
        'yProfile':yProfile,
        'yCOM':yCOM,
        'yRMS':yRMS,
        'yFWHM':yFWHM,
        'yCOMslice':yCOMslice,
        'yRMSslice':yRMSslice
    }
        
    return imageStats
    
def SubstractBackground(image,ROI,darkbg):
#Obtain all the statistics (profiles, center of mass, etc) of an image
#Arguments:
#   image: 2d numpy array where the first index correspond to y, and the second index corresponds to x
#   ROI: region of interest of the input image
#   darkbg: struct with the dark background image and its ROI
#Output
#   image: image after subtracting the background
#   ROI: region of interest of the ouput image
    
    #This only contemplates the case when the ROI of the darkbackground is larger than the ROI of the image. Other cases should be contemplated in the future
    minX=ROI['x0']-darkbg.ROI.x0;
    maxX=(ROI['x0']+ROI['xN']-1)-darkbg.ROI.x0;
    minY=ROI['y0']-darkbg.ROI.y0;
    maxY=(ROI['y0']+ROI['yN']-1)-darkbg.ROI.y0;    
    
    image=image-darkbg.darkbackground[minY:(maxY+1),minX:(maxX+1)]
        
    return image,ROI

    
def DenoiseImage(image,medianfilter,SNR_border):
#Get rid of some of the noise in the image (profiles, center of mass, etc) of an image
#Arguments:
#   image: 2d numpy array where the first index correspond to y, and the second index corresponds to x
#   medianfilter: number of neighbours for the median filter
#   SNR_border: number of pixels near the border that can be considered to contain just noise
#Output
#   image: filtered image
#   ok: true if there is something in the image

    ok=1
    #Applygin the median filter
    image=im.median_filter(image,medianfilter)
    
    #Obtaining the mean and the standard deviation of the noise
    mean=np.mean(image[:,0:SNR_border]);
    std=np.std(image[:,0:SNR_border]);
    
    #Subtracting the mean of the noise
    image=image-mean;
    
    #Setting a threshold equal to 10 times the standard deviation
    thres=10*std;    
    image[image < thres ] = 0 
    
    #We also normalize the image to have a total area of one
    if(np.sum(image)>0):
        image=image/np.sum(image)
    else:
        print 'Image Completely Empty'
        ok=0
        
    return image,ok

def FindROI(image,ROI,threshold,expandfactor):
#Find the subroi of the image
#Arguments:
#   image: 2d numpy array where the first index correspond to y, and the second index corresponds to x
#   ROI: region of interest of the input image
#   threshold: fraction of one that will set where the signal has dropped enough from the maximum to consider it to be the width the of trace
#   expandfactor: factor that will increase the calculated width from the maximum to where the signal drops to threshold
#Output
#   cropped: 3d numpy array with the cropped image where the first index always has one dimension (it will become the bunch index), the second index correspond to y, and the third index corresponds to x
#   outROI: region of interest of the output image

    #For the cropping on each direction we use the profile on each direction
    profileX=image.sum(0)
    profileY=image.sum(1)
    
    maxpos=np.argmax(profileX);                             #Position of the maximum
    thres=profileX[maxpos]*threshold;                       #Threshold value
    overthreshold=np.nonzero(profileX>=thres)[0];           #Indices that correspond to values higher than the threshold
    center=(overthreshold[0]+overthreshold[-1])/2;          #Middle position between the first value and the last value higher than th threshold
    width=(overthreshold[-1]-overthreshold[0])*expandfactor;  #Total width after applying the expand factor
    ind1X=np.round(center-width/2);                         #Index on the left side form the center
    ind2X=np.round(center+width/2);                         #Index on the right side form the center
    ind1X=np.amax([0,ind1X])                                #Check that the index is not too negative
    ind2X=np.amin([profileX.size,ind2X])                    #Check that the index is not too high
    
    #Same for y
    maxpos=np.argmax(profileY);
    thres=profileY[maxpos]*threshold;
    overthreshold=np.nonzero(profileY>=thres)[0];
    center=(overthreshold[0]+overthreshold[-1])/2;
    width=(overthreshold[-1]-overthreshold[0])*expandfactor;
    ind1Y=np.round(center-width/2);
    ind2Y=np.round(center+width/2);
    ind1Y=np.amax([0,ind1Y])
    ind2Y=np.amin([profileY.size,ind2Y])
        
    #Cropping the image using the calculated indices
    cropped=np.zeros((1,ind2Y-ind1Y,ind2X-ind1X))
    cropped[0,:,:]=image[ind1Y:ind2Y,ind1X:ind2X]
                
    #Output ROI in terms of the input ROI            
    outROI={
        'x0':ROI['x0']+ind1X,
        'y0':ROI['y0']+ind1Y,
        'xN':ind2X-ind1X+1,
        'yN':ind2Y-ind1Y+1,
        'x':ROI['x0']+np.arange(ind1X, ind2X),
        'y':ROI['y0']+np.arange(ind1Y, ind2Y)
        }
    
    return cropped,outROI

def ProcessLasingSingleShot(PU,imageStats,shotToShot,nolasingAveragedProfiles):
#Process a single shot profiles, using the no lasing references to retrieve the x-ray pulse(s)
#Arguments:
#   PU: physical units of the profiles
#   imageStats: statistics of the xtcav image (profiles)
#   shotToShot: structure with the shot information
#   nolasingAveragedProfiles: no lasing reference profiles
#Output
#   pulsecharacterization: retrieved pulse

    NB=len(imageStats)              #Number of bunches
    
    if (NB!=nolasingAveragedProfiles.NB[0,0]):
        print 'Different number of bunches in the reference'
        sys.exit()
    
    t=nolasingAveragedProfiles.t[0,:]   #Master time obtained from the no lasing references
    dt=(t[-1]-t[0])/(t.size-1)
    
    eCharge=1.60217657e-19          #Electron charge in coulombs
    Nelectrons=shotToShot['dumpecharge']/eCharge;   #Total number of electrons in the bunch    
    
    #Create the the arrays for the outputs, first index is always bunch number
    delay=np.zeros(NB, dtype=np.float64);                       #Delay from each bunch with respect to the first one in fs
    delaychange=np.zeros(NB, dtype=np.float64);                 #Difference between the delay from each bunch with respect to the first one in fs and the same form the non lasing reference
    energydiff=np.zeros(NB, dtype=np.float64);                  #Distance in energy for each bunch with respect to the first one in MeV
    energydiffchange=np.zeros(NB, dtype=np.float64);            #Comparison of that distance with respect to the no lasing
    eBunchCOM=np.zeros(NB, dtype=np.float64);                   #Energy of the XRays generated from each bunch for the center of mass approach in J
    eBunchRMS=np.zeros(NB, dtype=np.float64);                   #Energy of the XRays generated from each bunch for the dispersion of mass approach in J
    powerAgreement=np.zeros(NB, dtype=np.float64);              #Agreement factor between the two methods
    lasingECurrent=np.zeros((NB,t.size), dtype=np.float64);     #Electron current for the lasing trace (In #electrons/s)
    nolasingECurrent=np.zeros((NB,t.size), dtype=np.float64);   #Electron current for the no lasing trace (In #electrons/s)
    lasingECOM=np.zeros((NB,t.size), dtype=np.float64);         #Lasing energy center of masses for each time in MeV
    nolasingECOM=np.zeros((NB,t.size), dtype=np.float64);       #No lasing energy center of masses for each time in MeV
    lasingERMS=np.zeros((NB,t.size), dtype=np.float64);         #Lasing energy dispersion for each time in MeV
    nolasingERMS=np.zeros((NB,t.size), dtype=np.float64);       #No lasing energy dispersion for each time in MeV
    intensityECOM=np.zeros((NB,t.size), dtype=np.float64);      #Retrieved intensity in GeV based on ECOM
    intensityERMS=np.zeros((NB,t.size), dtype=np.float64);      #Retrieved intensity in GeV based on ERMS

        
     
    #If the step in time is negative, we mirror the x axis to make it ascending and consequently mirror the profiles     
    if PU['xfsPerPix']<0:
        PU['xfs']=PU['xfs'][::-1]
        for j in range(NB):
            imageStats[j]['xProfile']=imageStats[j]['xProfile'][::-1]
            imageStats[j]['yCOMslice']=imageStats[j]['yCOMslice'][::-1]
            imageStats[j]['yRMSslice']=imageStats[j]['yRMSslice'][::-1]
    
    #We treat each bunch separately
    for j in range(NB):
        distT=(imageStats[j]['xCOM']-imageStats[0]['xCOM'])*PU['xfsPerPix']  #Distance in time converted form pixels to fs
        distE=(imageStats[j]['yCOM']-imageStats[0]['yCOM'])*PU['yMeVPerPix'] #Distance in time converted form pixels to MeV
        
        delay[j]=distT  #The delay for each bunch is the distance in time
        energydiff[j]=distE #Same for energy
        
        dt_old=PU['xfs'][1]-PU['xfs'][0]; # dt before interpolation 
        
        eCurrent=imageStats[j]['xProfile']/(dt_old*1e-15)*Nelectrons                        #Electron current in number of electrons per second, the original xProfile already was normalized to have a total sum of one for the all the bunches together
        
        eCOMslice=(imageStats[j]['yCOMslice']-imageStats[j]['yCOM'])*PU['yMeVPerPix']       #Center of mass in energy for each t converted to the right units        
        eRMSslice=imageStats[j]['yRMSslice']*PU['yMeVPerPix']                               #Energy dispersion for each t converted to the right units

        interp=scipy.interpolate.interp1d(PU['xfs']-distT,eCurrent,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True)  #Interpolation to master time
        eCurrent=interp(t);    
                                                   
        interp=scipy.interpolate.interp1d(PU['xfs']-distT,eCOMslice,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True)  #Interpolation to master time
        eCOMslice=interp(t);
            
        interp=scipy.interpolate.interp1d(PU['xfs']-distT,eRMSslice,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True)  #Interpolation to master time
        eRMSslice=interp(t)        
        
        #Find best no lasing match
        NG=nolasingAveragedProfiles.eCurrent.shape[1];
        err= np.zeros(NG, dtype=np.float64);
        for g in range(NG):
            err[g] = np.corrcoef(eCurrent,nolasingAveragedProfiles.eCurrent[j,g,:])[0,1]**2;
        
        #The index of the most similar is that with a highest correlation, i.e. the last in the array after sorting it
        order=np.argsort(err)
        refInd=order[-1];
        
        #The change in the delay and in energy with respect to the same bunch for the no lasing reference
        delaychange[j]=distT-nolasingAveragedProfiles.distT[j,refInd]
        energydiffchange[j]=distE-nolasingAveragedProfiles.distE[j,refInd]
                                       
        #We do proper assignations
        lasingECurrent[j,:]=eCurrent
        nolasingECurrent[j,:]=nolasingAveragedProfiles.eCurrent[j,refInd,:]

        
        #We threshold the ECOM and ERMS based on electron current
        threslevel=0.1;
        threslasing=np.amax(lasingECurrent[j,:])*threslevel;
        thresnolasing=np.amax(nolasingECurrent[j,:])*threslevel;       
        indiceslasing=np.where(lasingECurrent[j,:]>threslasing)
        indicesnolasing=np.where(nolasingECurrent[j,:]>thresnolasing)      
        ind1=np.amax([indiceslasing[0][0],indicesnolasing[0][0]])
        ind2=np.amin([indiceslasing[0][-1],indicesnolasing[0][-1]])        
        if ind1>ind2:
            ind1=ind2
            
        
        #And do the rest of the assignations taking into account the thresholding
        lasingECOM[j,ind1:ind2]=eCOMslice[ind1:ind2]
        nolasingECOM[j,ind1:ind2]=nolasingAveragedProfiles.eCOMslice[j,refInd,ind1:ind2]
        lasingERMS[j,ind1:ind2]=eRMSslice[ind1:ind2]
        nolasingERMS[j,ind1:ind2]=nolasingAveragedProfiles.eRMSslice[j,refInd,ind1:ind2]
        
        
        #First calculation of the intensity based on center of masses and dispersion for each bunch
        intensityECOM[j,:]=((nolasingECOM[j,:]-lasingECOM[j,:])*eCharge*1e6)*eCurrent    #In J/s
        intensityERMS[j,:]=(lasingERMS[j,:]**2-nolasingERMS[j,:]**2)*(eCurrent**(2.0/3.0)) # In J/s
        
         
    #Calculate the normalization constants to have a total energy compatible with the energy detected in the gas detector
    eoffsetfactor=(shotToShot['xrayenergy']-(np.sum(intensityECOM)*dt*1e-15))/Nelectrons   #In J                           
    escalefactor=np.sum(intensityERMS)*dt*1e-15                 #in J
    
    
    #Apply the corrections to each bunch and calculate the final energy distribution and power agreement
    for j in range(NB):                 
        intensityECOM[j,:]=((nolasingECOM[j,:]-lasingECOM[j,:])*eCharge*1e6+eoffsetfactor)*lasingECurrent[j,:]*1e-9   #In GJ/s (GW)
        intensityERMS[j,:]=shotToShot['xrayenergy']*intensityERMS[j,:]/escalefactor*1e-9   #In GJ/s (GW)        
        powerAgreement[j]=1-np.sum((intensityECOM[j,:]-intensityERMS[j,:])**2)/(np.sum((intensityECOM[j,:]-np.mean(intensityECOM[j,:]))**2)+np.sum((intensityERMS[j,:]-np.mean(intensityERMS[j,:]))**2))
        eBunchCOM[j]=np.sum(intensityECOM[j,:])*dt*1e-15*1e9
        eBunchRMS[j]=np.sum(intensityERMS[j,:])*dt*1e-15*1e9
                    
    #Create the output structure
    pulsecharacterization={
        't':t,                                  #Master time vector in fs
        'intensityECOM':intensityECOM,          #Retrieved intensity in GeV based on ECOM
        'intensityERMS':intensityERMS,          #Retrieved intensity in GeV based on ERMS
        'powerAgreement':powerAgreement,        #Agreement between the two intensities
        'delay':delay,                          #Delay from each bunch with respect to the first one in fs
        'delaychange':delaychange,              #Difference between the delay from each bunch with respect to the first one in fs and the same form the non lasing reference
        'xrayenergy':shotToShot['xrayenergy'],  #Total x-ray energy from the gas detector in J
        'lasingenergyperbunchECOM': eBunchCOM,  #Energy of the XRays generated from each bunch for the center of mass approach in J
        'lasingenergyperbunchERMS': eBunchRMS,  #Energy of the XRays generated from each bunch for the dispersion approach in J
        'energydiff':energydiff,                #Distance in energy for each bunch with respect to the first one in MeV
        'energydiffchange':energydiffchange,    #Comparison of that distance with respect to the no lasing
        'lasingECurrent':lasingECurrent,        #Electron current for the lasing trace (In #electrons/s)
        'nolasingECurrent':nolasingECurrent,    #Electron current for the no lasing trace (In #electrons/s)
        'lasingECOM':lasingECOM,                #Lasing energy center of masses for each time in MeV
        'nolasingECOM':nolasingECOM,            #No lasing energy center of masses for each time in MeV
        'lasingERMS':lasingERMS,                #Lasing energy dispersion for each time in MeV
        'nolasingERMS':nolasingERMS,            #No lasing energy dispersion for each time in MeV
        'NB': NB                                #Number of bunches
        }
    
    
    return pulsecharacterization
    
def AverageXTCAVProfilesGroups(listROI,listImageStats,listShotToShot,globalCalibration,shotsPerGroup):
#Find the subroi of the image
#Arguments:
#   listROI: list with the axis for all the XTCAV non lasing profiles to average
#   listImageStats: list of the statistics (profiles) for all the XTCAV non lasing profiles to average
#   listShotToShot: list of the shot to shot properties structures for each profile
#   globalCalibration: global calibration structure for xtcav
#   shotsPerGroup
#Output
#   averagedProfiles: list with the averaged reference of the reference for each group 
   
    N=len(listImageStats)           #Total number of profiles
    NB=len(listImageStats[0])       #Number of bunches
    NG=int(np.floor(N/shotsPerGroup))   #Number of groups to make
            
    listPU=[]                       #List for the physical units for each profile
    
    # Obtain physical units and calculate time vector   
    maxt=0          #Set an initial maximum, minimum and increment value for the master time vector
    mint=0
    mindt=1000

    #For each profile obtain the physical units
    for i in range(N):
        PU=CalculatePhysicalUnits(listROI[i],[listImageStats[i][0]['xCOM'],listImageStats[i][0]['yCOM']],listShotToShot[i],globalCalibration)   
        #If the step in time is negative, we mirror the x axis to make it ascending and consequently mirror the profiles
        if PU['xfsPerPix']<0:
            PU['xfs']=PU['xfs'][::-1]
            for j in range(NB):
                listImageStats[i][j]['xProfile']=listImageStats[i][j]['xProfile'][::-1]
                listImageStats[i][j]['yCOMslice']=listImageStats[i][j]['yCOMslice'][::-1]
                listImageStats[i][j]['yRMSslice']=listImageStats[i][j]['yRMSslice'][::-1]                                               
        listPU.append(PU)      

        #We compare and update the maximum, minimum and increment value for the master time vector
        maxt=np.amax([maxt,np.amax(PU['xfs'])])
        mint=np.amin([mint,np.amin(PU['xfs'])])
        mindt=np.amin([mindt,np.abs(PU['xfsPerPix'])])
            
    #Obtain the number of electrons in each shot
    eCharge=1.60217657e-19          #Electron charge in coulombs
    Nelectrons=np.zeros(N, dtype=np.float64);
    for i in range(N): 
        Nelectrons[i]=listShotToShot[i]['dumpecharge']/eCharge
            
    #To be safe with the master time, we set it to have a step half the minumum step
    dt=mindt/2

    #And create the master time vector in fs
    t=np.arange(mint,maxt+dt,dt)
    
    #Create the the arrays for the outputs, first index is always bunch number, and second index is group number
    averageECurrent=np.zeros((NB,NG,len(t)), dtype=np.float64);       #Electron current in (#electrons/s)
    averageECOMslice=np.zeros((NB,NG,len(t)), dtype=np.float64);      #Energy center of masses for each time in MeV
    averageERMSslice=np.zeros((NB,NG,len(t)), dtype=np.float64);      #Energy dispersion for each time in MeV
    averageDistT=np.zeros((NB,NG), dtype=np.float64);                 #Distance in time of the center of masses with respect to the center of the first bunch in fs
    averageDistE=np.zeros((NB,NG), dtype=np.float64);                 #Distance in energy of the center of masses with respect to the center of the first bunch in MeV
    averageTRMS=np.zeros((NB,NG), dtype=np.float64);                  #Total dispersion in time in fs
    averageERMS=np.zeros((NB,NG), dtype=np.float64);                  #Total dispersion in energy in MeV
    
    #We treat each bunch separately, even group them separately
    for j in range(NB):
        #Decide which profiles are going to be in which groups and average them together
        #Calculate interpolated profiles in time for comparison
        profilesT=np.zeros((N,len(t)), dtype=np.float64);
        for i in range(N): 
            distT=(listImageStats[i][j]['xCOM']-listImageStats[i][0]['xCOM'])*listPU[i]['xfsPerPix']
            profilesT[i,:]=scipy.interpolate.interp1d(listPU[i]['xfs']-distT,listImageStats[i][j]['xProfile'],kind='linear',fill_value=0,bounds_error=False,assume_sorted=True)(t)
            
        #Decide of the groups based on correlation 
        group=np.zeros(N, dtype=np.int32);      #array that will indicate which group each profile sill correspond to
        group[:]=-1;                            #initiated to -1
        
        for g in range(NG):                     #For each group
            currRef=np.where(group==-1)[0]  
            currRef=currRef[0]                  #We pick the first member to be the first one that has not been assigned to a group yet

            group[currRef]=g;                   #We assign it the current group
            
            # We calculate the correlation of the first profile to the rest of available profiles
            err = np.zeros(N, dtype=np.float64);              
            for i in range(currRef,N): 
                if group[i] == -1:
                    err[i] = np.corrcoef(profilesT[currRef,:],profilesT[i,:])[0,1]**2;
                    
            #The 'shotsPerGroup-1' profiles with the highest correlation will be also assigned to the same group
            order=np.argsort(err)            
            for i in range(0,shotsPerGroup-1): 
                group[order[-(1+i)]]=g
                    
        #Once the groups have been decided, the averaging is performed
        for g in range(NG):                 #For each group
            for i in range(N):    
                if group[i]==g:             #We find the profiles that belong to that group and average them together
                
                    distT=(listImageStats[i][j]['xCOM']-listImageStats[i][0]['xCOM'])*listPU[i]['xfsPerPix'] #Distance in time converted form pixels to fs
                    distE=(listImageStats[i][j]['yCOM']-listImageStats[i][0]['yCOM'])*listPU[i]['yMeVPerPix'] #Distance in time converted form pixels to MeV
                    averageDistT[j,g]=averageDistT[j,g]+distT       #Accumulate it in the right group
                    averageDistE[j,g]=averageDistE[j,g]+distE       #Accumulate it in the right group
                    
                    averageTRMS[j,g]=averageTRMS[j,g]+listImageStats[i][j]['xRMS']*listPU[i]['xfsPerPix']   #Conversion to fs and accumulate it in the right group
                    averageERMS[j,g]=averageTRMS[j,g]+listImageStats[i][j]['yRMS']*listPU[i]['yMeVPerPix']  #Conversion to MeV and accumulate it in the right group
                                          
                    dt_old=listPU[i]['xfs'][1]-listPU[i]['xfs'][0]; # dt before interpolation   
                    eCurrent=listImageStats[i][j]['xProfile']/(dt_old*1e-15)*Nelectrons[i]                              #Electron current in electrons/s   
                    
                    eCOMslice=(listImageStats[i][j]['yCOMslice']-listImageStats[i][j]['yCOM'])*listPU[i]['yMeVPerPix'] #Center of mass in energy for each t converted to the right units
                    eRMSslice=listImageStats[i][j]['yRMSslice']*listPU[i]['yMeVPerPix']                                 #Energy dispersion for each t converted to the right units
                        
                    interp=scipy.interpolate.interp1d(listPU[i]['xfs']-distT,eCurrent,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True)  #Interpolation to master time                    
                    averageECurrent[j,g,:]=averageECurrent[j,g,:]+interp(t);  #Accumulate it in the right group                    
                                                
                    interp=scipy.interpolate.interp1d(listPU[i]['xfs']-distT,eCOMslice,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True) #Interpolation to master time
                    averageECOMslice[j,g,:]=averageECOMslice[j,g,:]+interp(t);          #Accumulate it in the right group
                    
                    interp=scipy.interpolate.interp1d(listPU[i]['xfs']-distT,eRMSslice,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True) #Interpolation to master time
                    averageERMSslice[j,g,:]=averageERMSslice[j,g,:]+interp(t);          #Accumulate it in the right group
                                  

            #Normalization off all the averaged stuff                      
            averageECurrent[j,g,:]=averageECurrent[j,g,:]/shotsPerGroup
            averageECOMslice[j,g,:]=averageECOMslice[j,g,:]/shotsPerGroup    
            averageERMSslice[j,g,:]=averageERMSslice[j,g,:]/shotsPerGroup
            averageDistT[j,g]=averageDistT[j,g]/shotsPerGroup
            averageDistE[j,g]=averageDistE[j,g]/shotsPerGroup
            averageTRMS[j,g]=averageTRMS[j,g]/shotsPerGroup
            averageERMS[j,g]=averageERMS[j,g]/shotsPerGroup     
    
    #Structure for the output
    averagedProfiles={
        't':t,                          #Master time in fs
        'eCurrent':averageECurrent,     #Electron current in (#electrons/s)
        'eCOMslice':averageECOMslice,   #Energy center of masses for each time in MeV
        'eRMSslice':averageERMSslice,   #Energy dispersion for each time in MeV
        'distT':averageDistT,           #Distance in time of the center of masses with respect to the center of the first bunch in fs
        'distE':averageDistE,           #Distance in energy of the center of masses with respect to the center of the first bunch in MeV
        'tRMS': averageTRMS,            #Total dispersion in time in fs
        'eRMS': averageERMS,            #Total dispersion in energy in MeV
        'NB': NB,                       #Number of bunches
        'NG': NG                        #Number of profiles
        }
            
    return averagedProfiles
    
def AverageXTCAVProfiles(listROI,listImageStats,listShotToShot,globalCalibration):

#Function that does the same as AverageXTCAVProfilesGroups without grouping, just averaging everything together, obsolete
    N=len(listImageStats)
    NB=len(listImageStats[0])
            
    listPU=[]
    
    
    maxt=0
    mint=0
    mindt=100

    for i in range(N):
        PU=CalculatePhysicalUnits(listROI[i],[listImageStats[i][0]['xCOM'],listImageStats[i][0]['yCOM']],listShotToShot[i],globalCalibration)             
        if PU['xfsPerPix']<0:
            PU['xfs']=PU['xfs'][::-1]
            for j in range(NB):
                listImageStats[i][j]['xProfile']=listImageStats[i][j]['xProfile'][::-1]
                listImageStats[i][j]['yCOMslice']=listImageStats[i][j]['yCOMslice'][::-1]
                listImageStats[i][j]['yRMSslice']=listImageStats[i][j]['yRMSslice'][::-1]
            
                                   
        listPU.append(PU)              
        maxt=np.amax([maxt,np.amax(PU['xfs'])])
        mint=np.amin([mint,np.amin(PU['xfs'])])
        mindt=np.amin([mindt,np.abs(PU['xfsPerPix'])])
                
    dt=mindt/2

    t=np.arange(mint,maxt+dt,dt)       
      
    averageTProfile=np.zeros((NB,len(t)), dtype=np.float64);
    averageECOMslice=np.zeros((NB,len(t)), dtype=np.float64);
    averageERMSslice=np.zeros((NB,len(t)), dtype=np.float64);
    averageDistT=np.zeros(NB, dtype=np.float64);
    averageDistE=np.zeros(NB, dtype=np.float64);
    averageTRMS=np.zeros(NB, dtype=np.float64);
    averageERMS=np.zeros(NB, dtype=np.float64); 
    
    for j in range(NB):
        for i in range(N):                
            distT=(listImageStats[i][j]['xCOM']-listImageStats[i][0]['xCOM'])*listPU[i]['xfsPerPix']
            distE=(listImageStats[i][j]['yCOM']-listImageStats[i][0]['yCOM'])*listPU[i]['yMeVPerPix']
            averageDistT[j]=averageDistT[j]+distT
            averageDistE[j]=averageDistE[j]+distE
            
            averageTRMS[j]=averageTRMS[j]+listImageStats[i][j]['xRMS']*listPU[i]['xfsPerPix']
            averageERMS[j]=averageTRMS[j]+listImageStats[i][j]['yRMS']*listPU[i]['yMeVPerPix']
             
            tProfile=listImageStats[i][j]['xProfile']
            eCOMslice=(listImageStats[i][j]['yCOMslice']-listImageStats[i][j]['yCOM'])*listPU[i]['yMeVPerPix']
            eRMSslice=listImageStats[i][j]['yRMSslice']*listPU[i]['yMeVPerPix']
                
            interp=scipy.interpolate.interp1d(listPU[i]['xfs']-distT,tProfile,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True)
            averageTProfile[j,:]=averageTProfile[j,:]+interp(t);
            #Renormalize so the total area is the same
            dt_old=listPU[i]['xfs'][1]-listPU[i]['xfs'][0];
            averageTProfile[j,:]=averageTProfile[j,:]+interp(t)*dt/dt_old;
                                        
            interp=scipy.interpolate.interp1d(listPU[i]['xfs']-distT,eCOMslice,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True)
            averageECOMslice[j,:]=averageECOMslice[j,:]+interp(t);
            
            interp=scipy.interpolate.interp1d(listPU[i]['xfs']-distT,eRMSslice,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True)
            averageERMSslice[j,:]=averageERMSslice[j,:]+interp(t);
            
                    
        averageTProfile[j,:]=averageTProfile[j,:]/N
        averageECOMslice[j,:]=averageECOMslice[j,:]/N    
        averageERMSslice[j,:]=averageERMSslice[j,:]/N
        averageDistT[j]=averageDistT[j]/N
        averageDistE[j]=averageDistE[j]/N
        averageTRMS[j]=averageTRMS[j]/N
        averageERMS[j]=averageERMS[j]/N
                
    
    averagedProfiles={
        't':t,
        'tProfile':averageTProfile,
        'eCOMslice':averageECOMslice,
        'eRMSslice':averageERMSslice,
        'distT':averageDistT,
        'distE':averageDistE,
        'tRMS': averageTRMS,
        'eRMS': averageERMS
        }
            
    return averagedProfiles
    
def SaveDarkBackground(path,accumulator,ROI,runs,n):
#Saves the dark background
#Arguments:
#   path: full file path
#   accumulator: accumulated images, without normalization
#   ROI: region of interest of the accumulator
#   runs: list if the runs used
#   n: total number of accumulated images

    output = {}
    output['dark'] = {
        'darkbackground' : accumulator/n,
        'ROI':ROI,
        'runs': runs,
        'n':n
    }
        
    scipy.io.savemat(path, output)
    
def SaveLasingData(path,listPulses,runs,n):
#Saves the list of the retrieved pulses
#Arguments:
#   path: full file path
#   listPulses: list of the retrieved pulses
#   runs: list if the runs used
#   n: total number of accumulated images
    output = {}
    output['lasing'] = {
        'pulses':listPulses,
        'runs':runs,
        'n':n
    }
    scipy.io.savemat(path, output)

def DebugSaveSet(path,ROI,image,imageStats):
#Saves an image with the retrieved statistics
#Arguments:
#   path: full file path
#   ROI: ROI of the image
#   image: image to be saved
#   imageStatistics: statistics to be saved
    output = {}
    output['imageset'] = {
        'ROI' : ROI,
        'image':image,
        'imageStats':imageStats,
    }
    
    scipy.io.savemat(path, output)
        
        
def SaveNoLasingData(path,averagedProfiles,runs,n):
#Saves the non lasing references
#Arguments:
#   path: full file path
#   averagedProfiles: groups of averaged profiles
#   runs: list if the runs used
#   n: total number of accumulated images
    output = {}
    output['nolasing'] = {
        'averagedProfiles' : averagedProfiles,
        'runs':runs,
        'n':n
    }

    scipy.io.savemat(path, output)
    
def CalculatePhysicalUnits(ROI,center,shotToShot,globalCalibration):
#Obtain the x axis and y axis in physical units (time and energy)
#Arguments:
#   ROI: region of interes containing the x and y axis
#   center: coordinates of the center of mass, array with two elements, the first one for the x coordinate and the second one for the y coordinate
#   shottoShot: structure with the properties of the specific shot
#   globalCalibration: structure with the global calibration for the xtcav
#Output:
#   outimage: 3d numpy array with the split image image where the first index is the bunch indesx, the second index correspond to y, and the third index corresponds to x
 
    umperpix=globalCalibration['umperpix']
    dumpe=globalCalibration['dumpe']
    dumpdisp=globalCalibration['dumpdisp']    
    rfampcalib=globalCalibration['rfampcalib']
    rfphasecalib=globalCalibration['rfphasecalib']    
    strstrength=globalCalibration['strstrength']
    
    rfamp=shotToShot['xtcavrfamp']
    rfphase=shotToShot['xtcavrfphase']

    yMeVPerPix = umperpix*dumpe/dumpdisp*1e-3;          #Spacing of the y axis in MeV
    
    xfsPerPix = -umperpix*rfampcalib/(0.3*strstrength*rfamp);     #Spacing of the x axis in fs (this can be negative)
    
    signflip = round(math.cos((rfphasecalib-rfphase)*math.pi/180)); #It may need to be flipped depending on the phase
    xfsPerPix = signflip*xfsPerPix;    
    
    xfs=xfsPerPix*(ROI['x']-center[0])                  #x axis in fs around the center of mass
    yMeV=yMeVPerPix*(ROI['y']-center[1])                #y axis in MeV around the center of mass
           
    
    physicalUnits={                                     #Structure with the physical units
        'yMeVPerPix':yMeVPerPix,
        'xfsPerPix':xfsPerPix,
        'xfs':xfs,
        'yMeV':yMeV
        }
           
           
    return physicalUnits

def GetGlobalXTCAVCalibration(epicsStore):
#Obtain the global XTCAV calibration form the epicsStore
#Arguments:
#   epicsStore
#Output:
#   globalCalibration: struct with the parameters
#   ok: if all the data was retrieved correctly

    ok=1
    umperpix=epicsStore.value('OTRS:DMP1:695:RESOLUTION')
    strstrength=epicsStore.value('OTRS:DMP1:695:TCAL_X')
    rfampcalib=epicsStore.value('SIOC:SYS0:ML01:AO214')
    rfphasecalib=epicsStore.value('SIOC:SYS0:ML01:AO215')
    dumpe=epicsStore.value('REFS:DMP1:400:EDES')
    dumpdisp=epicsStore.value('SIOC:SYS0:ML01:AO216')
    
    if umperpix==None:              #Some hardcoded values
        print 'No XTCAV Calibration'
        ok=0
        umperpix=35.47
        strstrength=48.2406251901
        rfampcalib=20
        rfphasecalib=180
        dumpdisp=0.56376747213
        dumpe=3.442
             
    globalCalibration={
        'umperpix':umperpix, #Pixel size of the XTCAV camera
        'strstrength':strstrength,  #Strength parameter
        'rfampcalib':rfampcalib,    #Calibration of the RF amplitude
        'rfphasecalib':rfphasecalib,    #Calibration of the RF phase
        'dumpe':dumpe,                  #Beam energy: dump config
        'dumpdisp':dumpdisp             #Vertical position to energy: dispersion
        }
                
                
    return globalCalibration,ok
          
    
def GetXTCAVImageROI(epicsStore):
#Obtain the ROI for the XTCAV image
#Arguments:
#   epicsStore
#Output:
#   ROI: struct with the ROI
#   ok: if all the data was retrieved correctly

    ok=1
    roiX=epicsStore.value('OTRS:DMP1:695:MinX')
    roiY=epicsStore.value('OTRS:DMP1:695:MinY')
    roiXN=epicsStore.value('OTRS:DMP1:695:SizeX')
    roiYN=epicsStore.value('OTRS:DMP1:695:SizeY')
    
    
    if roiX==None:           #Some hardcoded values
        print 'No XTCAV ROI info, no'
        ok=0
        roiX=0                                  
        roiY=0
        roiXN=1024
        roiYN=1024

    x=roiX+np.arange(0, roiXN)
    y=roiY+np.arange(0, roiYN)
    
    ROI={
        'x' :x,                 #X vector
        'y' :y,                 #Y vector
        'x0' :roiX,             #Position of the first pixel in x
        'y0' :roiY,             #Position of the first pixel in y
        'xN' :roiXN,            #Size of the image in X
        'yN' :roiYN};           #Size of the image in Y
           
    return ROI,ok
    
def ShotToShotParameters(ebeam,gasdetector):
#Obtain shot to shot parameters
#Arguments:
#   ebeam: psana.Bld.BldDataEBeamV5
#   gasdetector: psana.Bld.BldDataFEEGasDetEnergy
#Output:
#   ROI: struct with the ROI
#   ok: if all the data was retrieved correctly
    ok=1
    echarge=1.60217657e-19;

    if ebeam:    
        ebeamcharge=ebeam.ebeamCharge()
        xtcavrfamp=ebeam.ebeamXTCAVAmpl()
        xtcavrfphase=ebeam.ebeamXTCAVPhase()
        dumpecharge=ebeam.ebeamDumpCharge()*echarge #In C        
    else:    #Some hardcoded values
        print 'No ebeamv5 info'
        ok=0
        ebeamcharge=5
        xtcavrfamp=20
        xtcavrfphase=90
        energydetector=0.2;
        dumpecharge=175e-12 #In C
        
    if gasdetector:
        energydetector=(gasdetector.f_11_ENRC()+gasdetector.f_12_ENRC())/2    
    else:   #Some hardcoded values
        print 'No gas detector info'
        ok=0
        energydetector=0.2;
        
    energy=1e-3*energydetector #In J
            
    shotToShot={
        'ebeamcharge':ebeamcharge,  #ebeamcharge
        'dumpecharge':dumpecharge,  #dumpecharge in C
        'xtcavrfamp': xtcavrfamp,   #RF amplitude
        'xtcavrfphase':xtcavrfphase, #RF phase
        'xrayenergy':energy         #Xrays energy in J
        }        
       
    return shotToShot,ok               
    
def SplitImage(image, n):

#Split an XTCAV image depending of different bunches, this function is still to be programmed properly
#Arguments:
#   image: 3d numpy array with the image where the first index always has one dimension (it will become the bunch index), the second index correspond to y, and the third index corresponds to x
#   n: number of bunches expected to find
#Output:
#   outimage: 3d numpy array with the split image image where the first index is the bunch index, the second index correspond to y, and the third index corresponds to x
    
    
    if n==1:    #For one bunch, just the same image
        outimage=np.zeros((n,image.shape[1],image.shape[2]))
        outimage[0,:,:]=image[0,:,:]    
    elif n==2:  #For two bunches,  the image on the top and on the bottom of the center of mass
        
        #outimage=HorizontalLineSplitting(image[0,:,:]) 
        #outimage=OptimalSplittingLine(image[0,:,:])           

        outimage=IslandSplitting(image[0,:,:],2)
        
    else:       #In any other case just copies of the image, for debugging purposes
        outimage=IslandSplitting(image[0,:,:],n)
    
    return outimage
    
    
def HorizontalLineSplitting(image):
#Divides the image with a horizontal line at the center of mass
#Arguments:
#   image: 2d numpy array with the image where the first index correspond to y, and the second index corresponds to x
#Output:
#   outimage: 3d numpy array with the split image image where the first index is the bunch index, the second index correspond to y, and the third index corresponds to x
    
    outimage=np.zeros((2,image.shape[0],image.shape[1]))
    x=range(image.shape[1])
    y=range(image.shape[0])
    x0,y0=GetCenterOfMass(image[:,:],x,y)
    outimage[0,0:round(y0),:]=image[0:round(y0),:]
    outimage[1,round(y0):,:]=image[round(y0):,:]  
    
    return outimage
    
def RotatingLineSplitting(image):
#Divides the image with a straight line crossing the center of mass at the optimal angle
#Arguments:
#   image: 2d numpy array with the image where the first index correspond to y, and the second index corresponds to x
#Output:
#   outimage: 3d numpy array with the split image image where the first index is the bunch index, the second index correspond to y, and the third index corresponds to x
    
    outimage=np.zeros((2,image.shape[0],image.shape[1]))
    x=range(image.shape[1])
    y=range(image.shape[0])
    x0,y0=GetCenterOfMass(image[:,:],x,y)
    
    angles=np.linspace(-np.pi/2, np.pi/2, num=50, endpoint=False)
    
    areas=np.zeros(len(angles))
    cutinds = np.zeros((len(angles),len(x)))
        
    for i in range(len(angles)): 
        dx=np.cos(angles[i])
        dy=np.sin(angles[i])
                        
        if (angles[i]>=0):
            lengthleft=np.amin([y0/np.sin(angles[i]),x0/np.cos(angles[i])])
            lengthright=np.amin([(len(y)-y0-1)/np.sin(angles[i]),(len(x)-x0-1)/np.cos(angles[i])])
        else:
            lengthleft=np.amin([(len(y)-y0-1)/np.sin(-angles[i]),x0/np.cos(angles[i])])
            lengthright=np.amin([y0/np.sin(-angles[i]),(len(x)-x0-1)/np.cos(angles[i])])
            
            
        for j in range(-int(lengthleft),int(lengthright)):        
            areas[i]=areas[i]+image[int(np.round(j*dy+y0)),int(np.round(j*dx+x0))]

        
    

    
    ind=np.where(areas == np.amin(areas))

    #In case of more than one minimum value, better to take the one in the middle
    optimalangle=angles[int((ind[0][-1]+ind[0][0])/2)]            
            
    mask=np.zeros((len(y),len(x)))
    splitline=(x-x0)*np.tan(optimalangle)+y0 
    
    for i in range(len(x)): 
        mask[:,i]=y<splitline[i]
    
    outimage[0,:,:]=image*mask
    outimage[1,:,:]=image*(1-mask)
    
    return outimage
    
    
def OptimalSplittingLine(image):
#Find the optimal line to separate the two bunches and separates the image in two. Assuming they have different energy for each x value obtains the optimal y value between two peaks.
#Arguments:
#   image: 2d numpy array with the image where the first index correspond to y, and the second index corresponds to x
#Output:
#   outimage: 3d numpy array with the split image image where the first index is the bunch index, the second index correspond to y, and the third index corresponds to x
    #Use the center of mass as a starting point
    Nx=image.shape[1]
    Ny=image.shape[0]
    
    x=range(Nx)
    y=range(Ny)
    x0,y0=GetCenterOfMass(image,x,y)
    
    splitline=np.zeros(Nx, dtype=np.int32)
   
    
    x0=int(round(x0))
    y0=int(round(y0))
    
    #Filter the image along y axis
    fimage=im.filters.uniform_filter1d(image, 20, axis=0)
    
    #Each vertical slice of the right half of the image (left to right) from the x center of masses and then 
    #each vertical slice of the left half of the image (right to left) from the x center of masses  
    startPoint=y0
    for i in (range(x0,Nx)+range(x0-1,-1,-1)):
        #Find the point towards higher indices when the value starts to increase
        top=startPoint;    
        while (top<(Ny-1) and fimage[top,i]>=fimage[top+1,i]):
            top=top+1;    
        #Find the point towards lower indices when the value starts to increase
        bottom=startPoint;    
        while (bottom>0 and fimage[bottom,i]>=fimage[bottom-1,i]):
            bottom=bottom-1;
    
        #Use the middle value
        splitline[i]=int(round((top+bottom)/2))
        #The initial point for the next iteration will be the calculated point from he previous slice
        if (i!=(Nx-1)):
            startPoint=splitline[i];
        else: # except when we jump to the right half, in which case we we the value assigned for the initial point
            startPoint=splitline[x0];
        
        
    #For each x axis we split the slice
    outimage=np.zeros((2,image.shape[0],image.shape[1]))
    for i in range(image.shape[1]):
        outimage[0,0:splitline[i],i]=image[0:splitline[i],i]
        outimage[1,splitline[i]:,i]=image[splitline[i]:,i] 
        
    return outimage

    
def IslandSplitting(image,N):
#Find islands in the picture and order them by area, returning the image in N images, ordered by area. Teh total area is one.
#Arguments:
#   image: 2d numpy array with the image where the first index correspond to y, and thesecond index corresponds to x
#   N: number of islands to return
#Output:
#   outimages: 3d numpy array with the split image image where the first index is the bunch index, the second index correspond to y, and the third index corresponds to x
    
    #Use the center of mass as a starting point
    Nx=image.shape[1]
    Ny=image.shape[0]
    x=range(Nx)
    y=range(Ny)
    
    #Structure for the output
    outimages=np.zeros((N,Ny,Nx))
    
    #Get a bool image with just zeros and ones
    imgbool=image>0
    
    #Calculate the groups
    groups, n_groups =im.measurements.label(imgbool);
    
    #Structure for the areas and the images
    areas=np.zeros(n_groups,dtype=np.float64)
    images=[]
    
    #Obtain the separated images and the areas
    for i in range(0,n_groups):    
        images.append(image*(groups==(i+1)))
        areas[i]=np.sum(images[i])
        
    #Get the indices in descending area order
    orderareaind=np.argsort(areas)  
    orderareaind=np.flipud(orderareaind)
    
    #Number of valid images for the output
    n_valid=np.amin([N,n_groups])    
    
    #Calculate the angle of each large area island with respect to the center of mass
    x0,y0=GetCenterOfMass(image,x,y)        
    angles=np.zeros(n_valid,dtype=np.float64)
    for i in range(n_valid):
        xi,yi=GetCenterOfMass(images[orderareaind[i]],x,y)
        angles[i]=math.degrees(np.arctan2(yi-y0,xi-x0))
                
    #And we order the output based on angular distribution
    
    #If the distance of one of the islands to -180/180 angle is smaller than a certain fraction, we add an angle to make sure that nothing is close to the zero angle
    dist=180-abs(angles)
    if np.amin(dist)<30.0/n_valid:
        angles=angles+180.0/n_valid
            
    #Ordering
    orderangleind=np.argsort(angles)  
        
    #Assign the proper images to the output
    for i in range(n_valid):
        outimages[i,:,:]=images[orderareaind[orderangleind[i]]]
        
    #Renormalize to total area of 1
    outimages=outimages/np.sum(outimages)
    
    return outimages
    