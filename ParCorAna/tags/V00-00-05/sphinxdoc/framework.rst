
.. _framework:

################
 Documentation
################

Here we include more details on the using the Framework. 
Recall from the tutorial that the user writes a Python config file 
that defines two dictionaries

  * system_params
  * user_params

First we go over what to set in these dictionaries.

**************************
system_params
**************************
The system_params dictionary must define the following keys
  
  **dataset**
    what experiment/run to process, as well as index, live, or shared memory mode
  **src**, **psanaType**
    what detector data to scatter to the workers.
    determines the dimensions of NDArray in the domain of the user function **F**.
  **psanaOptions**, **outputArrayType**, **ndarrayProducerOutKey**, **ndarrCalibOutKey**
    all of these parameters define the psana configuration used. This is 
    intended to load and configure a calibration module.  The system provides a function for setting this up
    that uses the psana module ImgAlgos.NDArrCalib. Users can add options to the calibration, specify
    no calibration, or load their own modules here.
  **workerStoreDtype**
    to save space, workers could store float32 or possibly int16. They need to return
    results in float64.
  **maskNdarrayCoords** 
    a  mask file to exclude elements from worker calculation
  **testMaskNdarrayCoords** 
    a separate mask file for testing, should have no more than 100 pixels on.
  **numServers**
    how many servers to use in the system, most all other ranks are workers. 
    Typically 6 can keep up with live data. Use less for testing.
  **serverHosts**
    typically None, but can be set to specify hosts from which servers are chosen
  **times**
    how many times to store, the size of the **T** in the **T x NDArray** domain 
    for the function **F** (though the size of T is fixed, the times stored can change
    as the framework works through the data. T is a rolling buffer.
  **update**
    how frequently to have the workers calculate and update the viewer. This is in units of events.
    At 120 hz, set this to 12000 to publish results every 100 seconds.
  **delays**
    the **D** in the D x NDArray range for the function **F**. It is an integer array of 
    delay values, typically set on a logarithmic scale.
  **userClass**
    the Python class from which to instantiate the user module object. This must have all
    the callbacks described below. It carries out the correlation calculation.
  **h5output**
    output filename. Can be None in which case no h5output file is created.
  **testh5output** 
    output file for alternative test function of userClass when running the frameing in the alternate test mode.
  **overwrite**
    True if it is Ok to overwrite existing output file. 
  **verbosity**
    can print debug info if desired. Reports on progress of all MPI communication for each event.
  **numEvents**, **testNumEvents**
    for testing and doing shorter runs, one can set these to nonzero values.
  **elementsPerWorker**
    for testing and doing shorter runs, override mask and specify a few elements for each worker

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

The file

  ParCorAna/src/UserG2.py 

provides an example of carrying out the G2 calculation. It is carried out in three ways. One of these
is a testing method. The other two calculate G2 either during the update, or in an ongoing fashion.
This is implemented in three classes,

  **G2Common** 
    implements the testing function and common functionality to the ongoing and at end calculations
  **G2atEnd** 
    a straightforward calculation of the G2 terms right before the viewer is updated. Does O(T*n) work
    during viewer update, where T is the amount of data, and n is the number of delays
  **G2onGoing** 
    This keeps terms of the G2 calculation up to date as data comes in. It does O(n) work during each
    event, and O(T) work at the end.

The framework maintains a rolling buffer of the data. The size of this is controlled in the system_params. 
Default is to store 50,000 times and compute delays up to 25,000. If one processes data with 100,000 events, the
G2atEnd calculation will not include any of the data from the first 50,000 events, during the final update. However
the G2onGoing will.

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

