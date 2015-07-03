
.. _framework:

################
 Framework
################

Here we include more details on the Framework. 
The :ref:`configfile` section of the :ref:`tutorial` page covers all the details about the two dictionaries

  * system_params
  * user_params

that must be defined in the config file.

**************************
Mask File
**************************

* This needs to be a numpy array of an integral type saves to a .npy file.
* The shape of the array must be the same shape as what the psana ndarray producer module
  creates for the detector.  
* The values must be 0 or 1. 1 means this is a element of the detector to process.
  0 means don't process it.
* The tool parCorAnaMaskColorTool can help users produce this file, and convert from ndarray
  shape to image shape. Often a mask is created in image space, but it is best to create the color
  file and finecolor files in ndarray space and simply view them in image space.

.. _usercode:

************
WorkerData
************
The framework stores new data that is scattered from the servers to the workers.
It uses a class called WorkerData. WorkerData must provide an interface to the data
so users can quickly go through it in time order, and quickly look up data with a 
specific time delay or offset from a given piece of data. 

The framework maintains a rolling buffer of the data. The size of this is controlled 
in the system_params. Default is to store 50,000 times and compute delays up to 25,000. 

DAQ data can come out of order, and can be missing times, or have gaps, ie, one will
have data at the 120hz counter values of 1,2,4 but might never get 3. Moreover the data may
have arrived in the order 1,4,2. While the framework maintains a sorted list of the 
times, it does not move the detector data around. That is if one were to go through the
data array of WorkerData in order, you would look at data from 120hz counter times 1,4,2. 
However one does not do this, WorkerData provides an interface that lets you go through 
the times in order, and  look up specific times. It then gives you the correct index 
into the data array. 

We will see how to use this interface in the G2atEnd_, G2IncrementalAccumulator_, G2IncrementalWindowed_ 
sections below.

**************************
User Code
**************************

Recall that the framework assigns different roles to different ranks in the MPI world.
Users have an opportunity to run code in all roles except the master.
These roles are

  * **worker** most ranks will be workers. The framework stores data assigned to each worker, and the user code carries out the G2 
    calculation on this data.
  * **server** the servers get the calibrated detector data. User code may want to filter events based on other machine data, or
    preprocess the data.
  * **viewer** this is mostly user code as it is repsonible for publishing results. The framework will gather results from the
    workers for the viewer.
  * **master** the master coordinates which server scatters data to the workers and when the workers send results to the viewer.

The file::

  ParCorAna/src/UserG2.py 

provides an example of carrying out the G2 calculation. It is carried out in three ways and provides
a testing method. This is implemented in several classes covered in the sections 
G2atEnd_, G2IncrementalAccumulator_, G2IncrementalWindowed_ below.

Initialization
===================
A UserG2 class is initalized as follows::

    def __init__(self, user_params, system_params, mpiParams, testAlternate):

that is it takes the two params dict, mpiParams which includes a logger, and a flag to say if the 
alternative test is being done.


=====================
Callbacks
===================== 

The class you specify through system_params['userClass'] must provide a number of methods.
The framework will call these functions. Some are called only on server ranks, some only on worker ranks, 
and some only on the viewer rank. A separate instance of the userClass is created for each rank. A 
consequence of this is that modifications you make to an instance on a worker are not seen on the viewer.
For example, if one did::

  def serverCallback(self):
    self.badEvents = 3

  def workerCallback(self):
    print self.badEvents

you would get an error. serverCallback is only called on the server ranks. This will not add the attribute 
badEvents to instance of the userClass on the workers.

The framework handles the flow of all data between servers, workers and the viewer. It calls certain methods
by in the userClass after this data has been transferred, or before hand to decide if it should transfer data.

Presently, all of the below methods must be implemented in the userClass. Many will not be needed and can
be made optional in the future. For now though, a default implementation is provided in UserG2.py so users can
decide what they want to modify.UserG2.py has the most up to date documentation. 

All of these callbacks have names that start with either fw, server, worker, viewer. This indicates
which part of the framework calls the function. fw indiciates multiple roles use the function - i.e, both
workers and the viewer ranks in the framework will need to know how many arrays workers are calculating
and sending to the viewer.

fwArrayNames(self):
  returns names for the float64 arrays that are calculated. For the included UserG2 code, this returns
  ['G2', 'IF', 'IP']. This is an important function, the length of this list of names defines the N in the
  output of the user function. This is how the framework knows how much data to gather from workers for 
  viewers. It later uses these names to request data from the workers.

serverInit(self):
  called after framework initializes server. Rarely needed by user, however user's doing 
  custom calibration could load data they need for server processing here.

serverEventOk(self, evt): 
   look at the event, decide wether or not it should be processed. This is called before the
   framework extracts the detector data (which can take time). The intention is to look at other machine data, 
   like BeamLineData, to decide if this is an event one wants to process. Althouth it is possible to extract 
   the detector data here, there is another callback where users can examine the detector data after the 
   framework has extracted it (saves time not to extract it twice).
   
serverFinalDataArray(self, dataArray, evt): 
  if eventOk returns True, then the server roles of the framework extract the detector data.
  It is then passed to this user callback. If this callback returns None, presumably based on analyzing the 
  detector data, then the event is not processed. If dataArray is returned, or some other numpy array,
  then it is processed. Users can return a modified copy of dataArray. For instance, if one is doing 
  custom calibration that requires analysis of the entire detector image for common mode correction, 
  this is a place where one could do that. However workers also get a chance to adjust the data in
  the workerAdjustData function. 

workerInit(self, numELementsWorker):
  initializes worker. Each worker is told how many pixel elements of the detector it processes.
  This number can vary by at most one among the workers. G2Common creates the arrays that will be returend for
  G2, Ip and IF here - each being a numDelays x numElementsWorker array of float64. It also sets up the counts
  array and reads some user_params values that will be used during processing.

workerAdjustTerms(self, mode, dataIdx, pivotIndex, lenT, T, X):
  this function is used by G2onGoing, but not by G2atEnd. This lets workers adjust ongoing 
  terms in their final calculation based on new data. The parameters describe the new data coming in, and/or
  data being overwritten if the number of events has exceeded the times stored. This allows a class like
  G2atEnd to implement a windowed correlation analysis as well as a correlation analysis that covers the entire
  span of the data.

workerAdjustData(self, data):
  this is called before the framework stores data that workers will use for their correlation analysis.
  For example, if one wanted to set all non-positive numbers to a small value, each worker could execute
  that code on their portion of the data here. 

workerCalc(self, T, numTimesFilled, X): 
  this is an important function. This is called to create the final arrays that will be
  gathered from all the workers and sent to the viewer. This function returns a dictionary whose keys are
  the names returned by arrayNames, and whose values are the calculated arrays. It also returns counts of 
  how many pairs there are for each delay, as well as the int8array discussed in the overview to hold things
  like saturated pixels.

viewerInit(self, maskNdarrayCoords, h5GroupUser):
  called when the viewer is initialized. The viewer is responsible for binning results from the workers
  together as per the color file. However some of the pixels specified in the color file may be masked out. For 
  convenience, and to indicate that the viewer should use the mask, the read in mask file is passed to the viewer.
  The framework has also created (assuming system_params['h5output'] is not None) an h5output file and created a 
  group for the user results. The framework will save the system_params and user_params to the h5output file.
  The viewer is the only one who can write results of the calculation. It is intended that it write results into this
  group.

viewerPublish(counts, lastEventTime, name2delay2ndarray, int8array, h5UserGroup): 
  on viewer - this called after the results of all the workers have been gathered together. 
  It gets the counts, the timestamp and 120hz counter for the last event processed, 
  the gathered arrays, the gathered int array, and a h5py group into the h5output file to write to.
  For the UserG2 code, it will make use of the finecolor file in the users_params for its part in the calculation.

G2Common
============
This is a base or super class which does the following

* defines the array names, G2, IF and IP
* gets the delays
* provides a helper function calcAndPublishForTestAltHelper that is used by the super classes for the alternative test
* all of the viewer code/callbacks (gets color file in viewerInit, in viwerPublish, forms delay curves and either
  writes to the h5file, or plots).

G2atEnd
==========
This does all its work in workerCalc, O(T*D) work, where T is the number of Times, and D the number of delays.
Here is the code, showing how to work with the WorkerData class::

    def workerCalc(self, workerData):
        assert not workerData.empty(), "UserG2.workerCalc called on empty data"
        maxStoredTime = workerData.maxTimeForStoredData()
        
        for delayIdx, delay in enumerate(self.delays):
            if delay > maxStoredTime: break
            for tmA, xIdxA in workerData.timesDataIndexes():
                tmB = tmA + delay
                if tmB > maxStoredTime: break
                xIdxB = workerData.tm2idx(tmB)
                timeNotStored = xIdxB is None
                if timeNotStored: continue
                intensities_A = workerData.X[xIdxA,:]
                intensities_B = workerData.X[xIdxB,:]
                self.counts[delayIdx] += 1
                self.G2[delayIdx,:] += intensities_A * intensities_B
                self.IP[delayIdx,:] += intensities_A
                self.IF[delayIdx,:] += intensities_B

        return {'G2':self.G2, 'IP':self.IP, 'IF':self.IF}, self.counts, self.saturatedElements

G2IncrementalAccumulator
==========================
This does the G2 calculation by doing O(D) work on each event (where D is the number of delays). 
It does this by keeping the G2 calculation up to date when new data comes in Here is the main code::

    def workerAfterDataInsert(self, tm, xInd, workerData):
        maxStoredTime = workerData.maxTimeForStoredData()
        for delayIdx, delay in enumerate(self.delays):
            if delay > maxStoredTime: break
            tmEarlier = tm - delay
            xIndEarlier = workerData.tm2idx(tmEarlier)
            earlierLaterPairs=[]
            if xIndEarlier is not None:
                earlierLaterPairs.append((xIndEarlier, xInd))
            tmLater = tm + delay
            xIndLater = workerData.tm2idx(tmLater)
            if xIndLater is not None:
                earlierLaterPairs.append((xInd, xIndLater))
            for earlierLaterPair in earlierLaterPairs:
                idxEarlier, idxLater = earlierLaterPair
                intensitiesFirstTime = workerData.X[idxEarlier,:]
                intensitiesLaterTime = workerData.X[idxLater,:]
                self.G2[delayIdx,:] += intensitiesFirstTime * intensitiesLaterTime
                self.IP[delayIdx,:] += intensitiesFirstTime
                self.IF[delayIdx,:] += intensitiesLaterTime
                self.counts[delayIdx] += 1

G2IncrementalWindowed
======================
As new data comes in and overwrites old data, this removes the effect of the old data. 
It derives from G2IncrementalAccumulator, doing the same thing that it does during the
workerAfterDataInsert function, but it also overrides workerBeforeDataRemove as follows::

    def workerBeforeDataRemove(self, tm, xInd, workerData):
        maxStoredTime = workerData.maxTimeForStoredData()
        for delayIdx, delay in enumerate(self.delays):
            if delay > maxStoredTime: break
            earlierLaterPairs = []
            tmEarlier = tm - delay
            xIndEarlier = workerData.tm2idx(tmEarlier)
            if xIndEarlier is not None:
                earlierLaterPairs.append((xIndEarlier, xInd))
            tmLater = tm + delay
            xIndLater = workerData.tm2idx(tmLater)
            if xIndLater is not None:
                earlierLaterPairs.append((xInd, xIndLater))
            for earlierLaterPair in earlierLaterPairs:
                idxEarlier, idxLater = earlierLaterPair
                intensitiesEarlier = workerData.X[idxEarlier,:]
                intensitiesLater = workerData.X[idxLater,:]
                assert self.counts[delayIdx] > 0, "G2IncrementalWindowed.workerBeforeDataRemove - about to remove affect at delay=%d but counts=0" % delay
                self.counts[delayIdx] -= 1
                self.G2[delayIdx,:] -= intensitiesEarlier * intensitiesLater
                self.IP[delayIdx,:] -= intensitiesEarlier
                self.IF[delayIdx,:] -= intensitiesLater

**************************
Launching Jobs
**************************

See the section :ref:`runlocal` and :ref:`runonbatch` of the :ref:`tutorial` page 
for the basics.

To use some command line options, one could do

  mpiexec -n 4 parCorAnaDriver -c myconfig.py -v debug -n 300 -o myout.h5 --overwrite

To run against data on the ana file system in the psanaq, while saving the output to
a file, one could do:

  bsub -q psanaq -a mympi -n 30 -o g2calc_%J.out parCorAnaDriver -c myconfig.py -n 1000

