'''ParCorAna - this package provides a framework for doing cross 
correlation analysis using MPI to run calculations in parallel.

The framework does the following:
* gets detector data
* scatter it to workers
* gather results from workers
* manages an hdf5 output file

The user provides the following:
* a mask file identifying the detector elements to process
* a user module that:
** calculates correlations
** views or stores gathered results

The user module implements a multi valued function F. The domain of the function
is

  T x DetectorNDArray 

where T is a 120hz counter over a potentially large number of events and
DetectorNDArray represents an NDArray of the detector data. For example with CsPad2x2, this 
will be a rank 3 array with dimensions (185, 388, 2).

The output of the function has three parts. First, a number of arrays to hold
different parts of correlation analysis:

  D x DetectorNDArray1   (array datatype=float64)
  D x DetectorNDArray2   (array datatype=float64)
  ...
  D x DetectorNDArrayN   (array datatype=float64)

Here D represents a smaller number of outputs. For example, whereas T may be a counter 
for the last 50,000 events, D may be a set of delays on a log scale, 
for example D=[1,10,100,1000,10,000].  D is assumed to be in the same units as 
T, a 120hz counter.

The second part is counts for each of the delays:

  D x 1  delay counts    (array of int64)
                        
For this output counts, counts[d] is the number of pairs of times in T that are have a delay of
d between them. A counts entry can be zero for a delay that has not been seen yet in the data.

The third output for F is a general array

  DetetectorNDArray1     (array datatype=int8)
  
which applications can use for a variety of purposes. The intended application is
to record saturated pixels, pixels whose calibrated value that when over a threshold 
at some point and should thus be excluded from the presented results.

The parallelization is acheived by distributing the detector NDarray elements
among the workers. That is this framework is presently only for
functions F where the calculations across NDArray elements are independent
of one another. Each worker gets a fraction of the NDArray elements. These fractions
are shaped as flat arrays of float64.

To use this framework, the user does the following,

* tell the framework what dataset to process and what detector data to extract
* provides a mask file identifying what part of the detector data to process
* implement worker code to calculate F on subset of the detector data
* tells the framework how many output arrays it is computing
* implement viewer code to plot/save the D x NDArray* results
* launch an MPI job

== GETTING STARTED / TUTORIAL ==

* GET/BUILD SOFTWARE

To get started with the software, make a newrelease directory and add the ParCorAna
package to it (Presently the package is not part of the release). For example, do
these commands:

  newrel ana-current parCorRel
  cd parCorRel
  sit_setup
  # now get a kerboses ticket to get code from the repository
  kinit   
  addpkg ParCorAna HEAD
  scons

* MAKE CONFIG FILE

One first has to make a configuration file. This is a python file
that defines two dictionaries:

  system_params
  user_params

system_params defines a number of keys that the framework uses. user_params 
is only for the user module, the framework simple passes user_params through to the user module.

The simplest way to do this is to copy the default config file from the ParCorAna.
From the release directory parCorRel where you have added the ParCorAna package, do

  cp data/ParCorAna/default_params.py myconfig.py

You can read more about configuration in the section below and the comments in
the config file you have copied. Things you need to set are:

  system_params['dataset']
  system_params['src']    
  system_params['psanaType']

what dataset, source, and psana data type the framework needs to distribute to the workers.
The default values are for xpp tutorial data with full cspad. We will use these values
to run the tutorial.

  system_params['maskNdarrayCoords']

You need to provide the framework with a mask file for the dector data. This is a 
numpy array with the same dimensions as ndarray the psana calibration system uses to 
represent the detector. This is not neccessarily a 2D image. More on this below.

  system_params['times']
  system_params['delays']
  system_params['userClass']

For the tutorial, We will leave these alone. They specify a collection of 50,000 
events, and a set of 100 logarithmically spaced delays from 1 to 25,000. The user_class
is where users hook in their worker code. We will be using the example class in the ParCorAna
package - UserG2 - which does a simplified version of the G2 calculation used in XCS.

  user_params['colorNdarrayCoords']

this is a parameter that the UserG2 needs - a color file that labels the detector pixels
and determines which pixels are averaged together for the delay curve. It bins the pixels
into groups. More on this in the next section.

* CREATE MASK/COLOR FILE

The system requires a mask file that identifies the pixels to process. 
Reducing the number of pixels processed can be the key to fast feedback during an experiment.

The ParCorAna package provides a tool to make mask and color files in the numpy ndarray
format required. To read the tools help do

  parCorAnaMaskColorTool -h

(Another tool to look at is roicon, also part of the analysis release). The command

  parCorAnaMaskColorTool --start -d 'exp=xpptut13:run=1437' -t psana.CsPad.DataV2 -s 'DetInfo(XppGon.0:Cspad.0)' -n 300 -c 6

Will produce a mask and color file suitable for this tutorial:

  xpptut13-r1437_XppGon_0_Cspad_0_mask_ndarrCoords.npy  
  xpptut13-r1437_XppGon_0_Cspad_0_color_ndarrCoords.npy 

Note that our input will be ndarr files, not image files. The mask file is only  0 or 1. It is 1
for pixels that we INCLUDE. The color file uses 6 colors (since we gave the -c 6 option to the tool. 
As an example, these colors bin pixels based on intensity. In practice users will want to bin pixels
based on other criteria.

Once you have modified these files, or produced similarly formatted files you are ready for the 
next step.

* CONTINUE TO MODIFY CONFIG/PARMETERS FILE

Now in myconfig.py, set the mask and color file:

 system_params['maskNdarrayCoords'] = 'xpptut13-r1437_XppGon_0_Cspad_0_mask_ndarrCoords.npy'
 user_params['colorNdarrayCoords'] = 'xpptut13-r1437_XppGon_0_Cspad_0_color_ndarrCoords.npy'

Note that the last parameter is to the user_params - the framework knows nothing about the coloring.

Once you have modified the config file, it is a good idea to check that it runs as python code, i.e, that
all the imports work and the syntax is correct:

  python myconfig.py

The config file does a pretty-print of the two dictionaries defined.

* RUN THE SOFTWARE

Now you are ready to run the software. To test using a few cores on your local machine, do:

  mpiexec -n 4 parCorAnaDriver -c myconfig.py -n 100

This should run without error. You can get a listing of what is in the output file by doing

  h5ls -r g2calc_xpptut13-r1437.h5

The h5 file contains two groups at the root level:

  /system
  /user

In /system, one finds:

  /system/system_params    Dataset 
  /system/user_params      Dataset
  /system/color_ndarrayCoords Dataset
  /system/maskNdarrayCoords Dataset 

The first two are the output of the Python module pprint on the system_params and
user_params dictionaries after evaluating the config file.

The latter two are the mask and color ndarrays specified in the system_params.

In /user one finds whatever the user viewer code decides to write. The example 
UserG2 module writes, for example:

/user/G2_results_at_539  Group
/user/G2_results_at_539/G2 Group
/user/G2_results_at_539/G2/delay_000001 Dataset {32, 185, 388}
/user/G2_results_at_539/G2/delay_000002 Dataset {32, 185, 388}
...
/user/G2_results_at_539/IF Group
/user/G2_results_at_539/IF/delay_000001 Dataset {32, 185, 388}
/user/G2_results_at_539/IF/delay_000002 Dataset {32, 185, 388}
...
/user/G2_results_at_539/IP Group
/user/G2_results_at_539/IP/delay_000001 Dataset {32, 185, 388}
/user/G2_results_at_539/IP/delay_000002 Dataset {32, 185, 388}

== Configure Framework ==

Here we include more details on the configuration. Recall from the tutorial that the user writes a 
Python config file that defines two dictionaries:

  system_params
  user_params

* system_params defines
  - dataset  - what experiment/run to process, as well as index, live, or shared memory mode
  - src, type - what detector data to scatter to worker.
                determines the dimensions of the NDArray above
  - psana config - psana configuration, intended to load and configure a calibration module.
                   The system provides a function for setting this up.
                   Users may want to adjust the calibration options.
  - mask - a mask file to exclude elements from worker calculation
  - numservers - how many servers to use in the system, most all other ranks are workers
  - times - how many times to store, the size of the T in the T x NDArray domain 
            for the function F (though the size of T is fixed, the times stored can change
            as the framework works through the data. T is a rolling buffer.
  - update - how frequently to have the workers calculate and update the viewer
  - delays - the D in the D x NDArray range for the function F
  - user_class - the Python class from which to instantiate the user module object
  - h5output - output filename, default it to generate name based on dataset, but this
               doesn't work when running from shared memory.
  
* user_params defines any options the user module needs. For example with G2, a 
  color file to compute several delay curves for different nodes.

== provide a mask file ==
* This needs to be a numpy array of an integral type saves to a .npy file.
* The shape of the array must be the same shape as what the psana ndarray producer module
  creates for the detector.  
* The values must be 0 or 1. 1 means this is a element of the detector to process.
  0 means don't process it.
* The tool parCorAnaMaskColorTool can help users produce this file, and convert from ndarray
  shape to image shape. Often a mask is created in image space, but it is best to create the color
  file in ndarray space and simply view it in image space.

== Implement User Code ==

You can modify

  ParCorAna/src/UserG2.py 

to get started.  UserG2.py is a basic example which does the G2 calculation.
Note that the __init__ function receives the user_params dict (as well as system_params)
Note though, the class is called on server, worker and viewer ranks. Users can defer
initialization specific to those ranks in separate callback functions.

Here is a summary of the callbacks:

* MINIMUM SET OF CALLBACKS TO IMPLEMENT

** fwArrayNames(self):   returns names for the float64 arrays calculated: ['G2', 'IF', 'IP']
** workerInit(self, numELementsWorker): this is where the user code is told how many elements
   of the detector NDArray this worker will process.
** workerCalc(self, T, numTimesFilled, X): calculate the portion of the G2, IF and IP ndarrays 
   that this worker is responsible for,  as well as the corresponding int8array, and counts arrays
   for this portion. Returns all of these to the framework.
** viewerPublish(counts, relsec, name2delay2ndarray, int8array, h5UserGroup): this is
   called when the results have been gathered from the workers. The user computes the final
   answer and can write it in the hdf5 group, or plot it. A summary of the arguments:
   - name2delay2ndarray:  a 2D dictionary to get at the gathered ndarrays. For example
                          name2delay2ndarray['G2'][3] would return the ndarray for delay 3 
                          of the G2 term, gathered from all the workers.
   - int8array:           gathered from all the workers and has the ndarray shape. 
   - counts:              has only been received from the first worker. It is assumed that all 
                          workers produce the same counts array.

* OPTIONAL BUT USEFUL CALLBACKS
** serverEventOk(self, evt): look at the event, decide whether or not workers should process it.
   The intention is that the user only look at small things before the framework takes the
   time to extract the detector data.
* workerAdjustData(self, data): allows workers to adjust the data that the framework will store,
   on a per event basis. Examples might be removing true zero's, or identifying saturated 
   pixels based on a threshold.
* viewerInit(self, maskNdarrayCoords, h5GroupUser): allows viewer to initialize. Viewer is passed in 
  loaded mask array, as a logical index array - True for elements of the ndarray to include.
  Viewer also receives h5py group to write to. This group will be passed in the publish function as well.

* OPTIONAL AND USUALLY NOT NEEDED CALLBACKS 
* serverInit(self):  
* serverFinalDataArray(self, dataArray, evt): this is an opportunity for the user to filter the
  event by looking at the data array as well as other event data. This function returns None
  to filter, or the passed in dataArray to proceed. The function can also return modified copy
  of the array.
* workerAdjustTerms(self, mode, dataIdx, T, X): this allows workers to implement a 'rolling' calculation
  based on knowing when new data is added and replaces the oldest data

== launch an MPI job ==

* For testing locally 

mpiexec -n 4 parCorAnaDriver -c myconfig.py

* To use some command line options, 

mpiexec -n 4 parCorAnaDriver -c myconfig.py -v debug -n 300 -o myout.h5 --overwrite

* To run in the offline batch job:

bsub -q psanaq -a mympi -n 30 -o g2calc_%J.out parCorAnaDriver -c myconfig.py -n 1000

* To run in live mode, or shared memory mode
TODO

== Architechture ==

Below we take a top down approach to summarizing the components of the framework

----------------
* mpiFnDriver  - this is where the system starts. Key steps that it takes:
**  reads the config file, command line options override config with 
    verbosity, numevents, h5output, elementsperworker
** framework = CommSystemFramework(system_params, user_params)
   framework.run()

--------------
* CommSystemFramework - this is what the mpiFnDriver kicks off. This handles the mpi
   communication between master/servers/workers/viewer. It is meant to be agnostic of
   the kind of calculation being done. Key steps:
** identifyServerRanks - looks at numservers, identifies server ranks.
** identifyCommSubsystems - splits ranks into servers, workers, viewer, master. 
                     Creates intra-communicators for collective communication between
                          viewer <-> workers
                          each server <-> workers
** loads mask file
** Creates XCorrBase - part of the framework. The CommSystem talks to the XCorrBase.
   The XCorrBase talks to the user module. 
   XCorrBase knows about the kind of function F being implemented. It handles 
*** splitting ndarrays among the workers, sending them flattened 1D arrays of elements
*** gathering results
*** delivering and gathered results ad reassembled NDArrays to viewer code

* runCommSystem - this does different things depending on whether or not the rank is
                  a server, worker, viewer, or the master. We describe this below:
- - - - - - -
** Server
*** XCorrBase.serverInit() -> in turn calls userObj.serverInit()
*** runServer = RunServer() # creates a RunServer object (in the CommSystem)
*** runSurver.run()
- - - - - - -
** Worker

- - - - - - -
** Viewer
*** XCorrBase.viewerInit() -> see below, in turn calls userObj.viewerInit()

- - - - - - -
** Master
*** runMaster = RunMaster() # creates a RunMaster object (in the CommSystem)
*** runMaster.run()

------------
* XCorrBase -
** init: 
*** self.userObj = creates instance of user module from class in system_params['user_class']
** viewerInit()
*** 
***  calls self.userObj.fwArrayNames(): users code implements this function. This is
     how framework knows how many arrays user F function is computing    
'''
from CommSystem import identifyCommSubsystems, identifyServerRanks
from CommSystem import RunServer, RunMaster, RunWorker, RunViewer
from CommSystem import runCommSystem
from CommSystem import CommSystemFramework
from CommSystemUtil import checkCountsOffsets, divideAmongWorkers, makeLogger, checkParams, formatFileName, imgBoundBox
from MessageBuffers import SM_MsgBuffer, MVW_MsgBuffer
from PsanaUtil import parseDataSetString, makePsanaOptions, psanaNdArrays
from PsanaUtil import readDetectorDataAndEventTimes, getSortedCountersBasedOnSecNsecAtHertz
from XCorrBase import makeDelayList, writeToH5Group, XCorrBase, writeConfig
from WorkerData import WorkerData
import maskColorImgNdarr
from Exceptions import *

__all__ = ['SM_MsgBuffer', 'MVW_MsgBuffer', 
           'RunServer', 'RunMaster', 'RunViewer', 'RunWorker',
           'identifyCommSubsystems','identifyServerRanks',
           'runCommSystem', 'CommSystemFramework', 'maskColorImgNdarr',
           'checkCountsOffsets', 'divideAmongWorkers', 'makeLogger',
           'divideAmongWorkers', 'checkCountsOffsets',
           'WorkerData', 'XCorrBase', 'makeDelayList', 'writeToH5Group',
            'checkParams', 'formatFileName', 'imgBoundBox']
