
.. _framework:

################
 Documentation
################

Here we include more details on the using the Framework. 
Recall from the tutorial that the user writes a Python config file 
that defines two dictionaries::

  system_params
  user_params

**************************
system_params
**************************
system_params defines
  
  dataset
    what experiment/run to process, as well as index, live, or shared memory mode
  src, type
    what detector data to scatter to worker.
    determines the dimensions of the NDArray above
  psana config
    psana configuration, intended to load and configure a calibration module.
    The system provides a function for setting this up.
    Users may want to adjust the calibration options.
  mask
    a mask file to exclude elements from worker calculation
  numservers
    how many servers to use in the system, most all other ranks are workers
  times
    how many times to store, the size of the T in the T x NDArray domain 
    for the function F (though the size of T is fixed, the times stored can change
    as the framework works through the data. T is a rolling buffer.
  update
    how frequently to have the workers calculate and update the viewer
  delays
    the D in the D x NDArray range for the function F
  user_class
    the Python class from which to instantiate the user module object
  h5output
    output filename, default it to generate name based on dataset, but this
    doesn't work when running from shared memory.
  
**************************
user_params
**************************
user_params defines any options the user module needs. For example with G2, a 
color file to compute several delay curves for different nodes.



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
  file in ndarray space and simply view it in image space.

**************************
User Code
**************************

You can modify

  ParCorAna/src/UserG2.py 

to get started.  UserG2.py is a basic example which does the G2 calculation.
Note that the __init__ function receives the user_params dict (as well as system_params)
Note though, the class is called on server, worker and viewer ranks. Users can defer
initialization specific to those ranks in separate callback functions.

=====================
Basic Callbacks
===================== 

arrayNames(self):
  returns names for the float64 arrays calculated: ['G2', 'IF', 'IP']

initWorker(self, numELementsWorker):
  this is where the user code is told how many elements
  of the detector NDArray this worker will process.

workerCalc(self, T, numTimesFilled, X): 
  calculate the portion of the G2, IF and IP ndarrays 
  that this worker is responsible for,  as well as the corresponding int8array, and counts arrays
  for this portion. Returns all of these to the framework.

viewerPublish(counts, relsec, name2delay2ndarray, int8array, h5UserGroup): 
  this is called when the results have been gathered from the workers. The user computes the final
  answer and can write it in the hdf5 group, or plot it. A summary of the arguments::

    name2delay2ndarray:  a 2D dictionary to get at the gathered ndarrays. For example
                          name2delay2ndarray['G2'][3] would return the ndarray for delay 3 
                          of the G2 term, gathered from all the workers.
    int8array:           gathered from all the workers and has the ndarray shape. 
    counts:              has only been received from the first worker. It is assumed that all 
                          workers produce the same counts array.


=====================
Optional Callbacks
===================== 

eventOk(self, evt): look at the event, decide whether or not workers should process it.
   The intention is that the user only look at small things before the framework takes the
   time to extract the detector data.

adjustData(self, data): 
  allows workers to adjust the data that the framework will store,
  on a per event basis. Examples might be removing true zero's, or identifying saturated 
  pixels based on a threshold.

initViewer(self, mask_ndarrayCoords, h5GroupUser): 
  allows viewer to initialize. Viewer is passed in 
  loaded mask array, as a logical index array - True for elements of the ndarray to include.
  Viewer also receives h5py group to write to. This group will be passed in the publish function as well.

initServer(self):  
  server intialization

finalDataArray(self, dataArray, evt): 
  this is an opportunity for the user to filter the
  event by looking at the data array as well as other event data. This function returns None
  to filter, or the passed in dataArray to proceed. The function can also return modified copy
  of the array.

adjustTerms(self, mode, dataIdx, T, X): 
  this allows workers to implement a 'rolling' calculation
  based on knowing when new data is added and replaces the oldest data


**************************
Launching Jobs
**************************

For testing locally 

  mpiexec -n 4 parCorAnaDriver -c myconfig.py

To use some command line options, 

  mpiexec -n 4 parCorAnaDriver -c myconfig.py -v debug -n 300 -o myout.h5 --overwrite

To run in the offline batch job:

  bsub -q psanaq -a mympi -n 30 -o g2calc_%J.out parCorAnaDriver -c myconfig.py -n 1000

To run in live mode, or shared memory mode

  TODO

