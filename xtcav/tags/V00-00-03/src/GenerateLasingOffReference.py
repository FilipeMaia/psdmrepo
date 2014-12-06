#(c) Coded by Alvaro Sanchez-Gonzalez 2014

import os
import time
import psana
import numpy as np
import glob
import pdb
import IPython
import sys
import getopt
import warnings
import Utils as xtu
from DarkBackground import *
from LasingOffReference import *
from CalibrationPaths import *

class GenerateLasingOffReference(object):
    """
    Class that generates a set of lasing off references for XTCAV reconstruction purposes
    Attributes:
        experiment (str): String with the experiment reference to use. E.g. 'amoc8114'
        runs (str): String with a run number, or a run interval. E.g. '123'  '134-156' 145,136'
        maxshots (int): Maximum number of images to use for the references.
        calibrationpath (str): Custom calibration directory in case the default is not intended to be used.
        nb (int): Number of bunches.
        medianfilter (int): Number of neighbours for median filter.
        snrfilter (float): Number of sigmas for the noise threshold.
        groupsize (int): Number of profiles to average together for each reference.
        roiwaistthres (float): ratio with respect to the maximum to decide on the waist of the XTCAV trace.
        roiexpand (float): number of waists that the region of interest around will span around the center of the trace.

    """
        


    def __init__(self):
    
        #Handle warnings
        warnings.filterwarnings('always',module='Utils',category=UserWarning)
        warnings.filterwarnings('ignore',module='Utils',category=RuntimeWarning, message="invalid value encountered in divide")
    
        #Some default values for the options
        self._experiment='amoc8114'     #Experiment label
        self._maxshots=401              #Maximum number of valid shots to process
        self._runs='86'                 #Runs
        self._validityrange=[]
        self._darkreferencepath=[];         #Dark reference information
        self._nb=1                      #Number of bunches
        self._groupsize=5               #Number of profiles to average together
        self._medianfilter=3            #Number of neighbours for median filter
        self._snrfilter=10              #Number of sigmas for the noise threshold
        self._roiwaistthres=0.2         #Parameter for the roi location
        self._roiexpand=2.5             #Parameter for the roi location
        self._calpath=''

    def Generate(self):
        """
        After setting all the parameters, this method has to be called to generate the lasing off reference and save it in the proper location. It not set, the validity range for the reference will go from the first run number used to generate the reference and the last run.
        """
        print 'Lasing off reference'
        print '\t Experiment: %s' % self._experiment
        print '\t Runs: %s' % self._runs
        print '\t Number of bunches: %d' % self._nb
        print '\t Valid shots to process: %d' % self._maxshots
        print '\t Dark reference run: %s' % self._darkreferencepath

        #Loading the data, this way of working should be compatible with both xtc and hdf5 files
        dataSource=psana.DataSource("exp=%s:run=%s" % (self._experiment,self._runs))

        #Camera and type for the xtcav images
        xtcav_camera = psana.Source('DetInfo(XrayTransportDiagnostic.0:Opal1000.0)')
        xtcav_type=psana.Camera.FrameV1

        #Ebeam type: it should actually be the version 5 which is the one that contains xtcav stuff
        ebeam_data=psana.Source('BldInfo(EBeam)')

        #Gas detectors for the pulse energies
        gasdetector_data=psana.Source('BldInfo(FEEGasDetEnergy)')

        #Stores for environment variables   
        epicsStore = dataSource.env().epicsStore();

        n=0 #Counter for the total number of xtcav images processed

        #Empty lists for the statistics obtained from each image, the shot to shot properties, and the ROI of each image (although this ROI is initially the same for each shot, it becomes different when the image is cropped around the trace)
        listImageStats=[];
        listShotToShot=[];
        listROI=[];
 
        runs=numpy.array([],dtype=int) #Array that contains the run processed run numbers
        #for r,run in enumerate(dataSource.runs()):
        for r in [0]: 
            run=dataSource.runs().next(); #This line and the previous line are a temporal hack to go only through the first run, that avoids an unexpected block when calling next at the iterator, when there are not remaining runs.
            runs = numpy.append(runs,run.run());
            n_r=0 #Counter for the total number of xtcav images processed within the run        
            for e, evt in enumerate(run.events()):
                ebeam = evt.get(psana.Bld.BldDataEBeamV6,ebeam_data)        
                if not ebeam:
                    ebeam = evt.get(psana.Bld.BldDataEBeamV5,ebeam_data)
                
                gasdetector=evt.get(psana.Bld.BldDataFEEGasDetEnergy,gasdetector_data) 
                
                #After the first event the epics store should contain the ROI of the xtcav images and the calibration of the XTCAV
                if not 'ROI_XTCAV' in locals(): 
                    ROI_XTCAV,ok=xtu.GetXTCAVImageROI(epicsStore) 
                    if not ok: #If the information is not good, we try next event
                        del ROI_XTCAV
                        continue
                if not 'globalCalibration' in locals():
                    globalCalibration,ok=xtu.GetGlobalXTCAVCalibration(epicsStore)
                    if not ok: #If the information is not good, we try next event
                        del globalCalibration
                        continue
                #If we have not loaded the dark background information yet, we do
                if not 'db' in locals():
                    if not self._darkreferencepath:
                        cp=CalibrationPaths(dataSource,self._calpath)       
                        self._darkreferencepath=cp.findCalFileName('pedestals',evt.run())
                        
                    if not self._darkreferencepath:
                        print ('Dark reference for run %d not found, image will not be background substracted' % evt.run())
                        self._loadeddarkreference=False 
                        db=False
                    else:
                        db=DarkBackground.Load(self._darkreferencepath)
        
                    
                
                              
                frame = evt.get(xtcav_type, xtcav_camera) 
                if frame: #For each shot that contains an xtcav frame we retrieve it        
                    img=frame.data16().astype(np.float64)
                    
                    if np.max(img)>=16383 : #Detection if the image is saturated, we skip if it is
                        print 'Saturated Image'
                        continue
                        
                    shotToShot,ok = xtu.ShotToShotParameters(ebeam,gasdetector) #Obtain the shot to shot parameters necessary for the retrieval of the x and y axis in time and energy units
                    if not ok: #If the information is not good, we skip the event
                        continue
                    
                    #Subtract the dark background, taking into account properly possible different ROIs, if it is available
                    if db:        
                        img,ROI=xtu.SubtractBackground(img,ROI_XTCAV,db.image,db.ROI) 
                    else:
                        ROI=ROI_XTCAV
                    img,ok=xtu.DenoiseImage(img,self._medianfilter,self._snrfilter)                    #Remove noise from the image and normalize it
                    if not ok:                                        #If there is nothing in the image we skip the event  
                        continue
                    img,ROI=xtu.FindROI(img,ROI,self._roiwaistthres,self._roiexpand)                  #Crop the image, the ROI struct is changed. It also add an extra dimension to the image so the array can store multiple images corresponding to different bunches
                    img=xtu.SplitImage(img,self._nb)
                    imageStats=xtu.ProcessXTCAVImage(img,ROI)          #Obtain the different properties and profiles from the trace                                                                                                                  
                                                                     
                    listImageStats.append(imageStats)
                    listShotToShot.append(shotToShot)
                    listROI.append(ROI)
                    
                    n=n+1
                    n_r=n_r+1            
                    print "%d/%d" % (n_r,self._maxshots) #Debugging purposes, will be removed             
                                                                  
                if n_r>=self._maxshots: #After a certain number of shots we stop (Ideally this would be an argument, rather than a hardcoded value)
                    break
               
        #At the end, all the reference profiles are converted to Physical units, grouped and averaged together
        averagedProfiles = xtu.AverageXTCAVProfilesGroups(listROI,listImageStats,listShotToShot,globalCalibration,self._groupsize);     

        lor=LasingOffReference()
        lor.averagedProfiles=averagedProfiles
        lor.runs=runs
        lor.n=n
        
        parameters= {
            'version' : 0,
            'darkreferencepath':self._darkreferencepath,
            'nb':self._nb,
            'groupsize':self._groupsize,
            'medianfilter':self._medianfilter,
            'snrfilter':self._snrfilter,
            'roiwaistthres':self._roiwaistthres,
            'roiexpand':self._roiexpand
        }
        
        lor.parameters=parameters
        
        if not self._validityrange:
            validityrange=[runs[0], 'end']
        else:
            validityrange=self._validityrange
            
        cp=CalibrationPaths(dataSource,self._calpath)
        file=cp.newCalFileName('lasingoffreference',validityrange[0],validityrange[1])
                
        lor.Save(file)
        
        
    def SetValidityRange(self,runBegin,runEnd='end'):
        """Sets the validity range for the generated reference.

        Args:
            runBegin (int): First run in the range.
            runEnd (int or str): Last run in the range (use 'end' to leave the range open)..

        """
        self._validityrange=[runBegin, runEnd];
        
    #Access to the different properties: here we can change flags when a parameter is changed, or check the validity of the property
    @property
    def maxshots(self):
        return self._maxshots
    @maxshots.setter
    def maxshots(self, maxshots):
        self._maxshots = maxshots  
    @property            
    def experiment(self):
        return self._experiment
    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
    @property
    def runs(self):
        return self._runs
    @runs.setter
    def runs(self, runs):
        self._runs = runs
    @property
    def nb(self):
        return self._nb
    @nb.setter
    def nb(self, nb):
        self._nb = nb
    @property
    def medianfilter(self):
        return self._medianfilter
    @medianfilter.setter
    def medianfilter(self, medianfilter):
        self._medianfilter = medianfilter
    @property
    def snrfilter(self):
        return self._snrfilter
    @snrfilter.setter
    def snrfilter(self, snrfilter):
        self._snrfilter = snrfilter
    @property
    def groupsize(self):
        return self._groupsize
    @groupsize.setter
    def groupsize(self, groupsize):
        self._groupsize = groupsize
    @property
    def roiwaistthres(self):
        return self._roiwaistthres
    @roiwaistthres.setter
    def roiwaistthres(self, roiwaistthres):
        self._roiwaistthres = roiwaistthres
    @property
    def roiexpand(self):
        return self._roiexpand
    @roiexpand.setter
    def roiexpand(self, roiexpand):
        self._roiexpand = roiexpand
    @property
    def calibrationpath(self):
        return self._calpath
    @calibrationpath.setter
    def calibrationpath(self, calpath):
        self._calpath = calpath
    
