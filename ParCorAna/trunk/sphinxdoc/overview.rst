
.. _overview:

##########
 Overview
##########

The framework does the following:
  * gets detector data
  * scatters it to workers
  * gather results from workers
  * manages an hdf5 output file

The framework identifies different mpi ranks as
  * workers - these handle a subset of the detector pixels and 
    call user code to compute correlations for each pixel over time
  * servers - these get data using psana and call user code to filter
    and pre-process events
  * viewer - consolidates correlation results from workers, calls user
    code to plots or save results to disk
  * master - synchronizes various parts of the framework, no user code involved

The user provides the following:
  * a mask file identifying the detector elements to process
  * a user module that runs on servers, workers and the viewer. This module:

    * calculates correlations for the pixels assigned to each worker
    * plots or saves gathered results in the viewer
    * optionally filters/pre-processes events in the servers
    * optionally provides a testing mask file, and testing function


**********************
 User Function Domain
**********************
The user module implements a multi-valued function **F**. 
The domain of the function is::

  T x DetectorNDArray

where 

**T**
  is a 120hz counter over a potentially large number of events. This counter is derived from the 
  timestamp and fiducials for the events psana sees. The first event gets counter value 0. 
  If the DAQ produces out of order events (a rare damage condition) the framework can assign a 
  negative counter value to an event.

**DetectorNDArray**
  represents an NDArray of the detector data. 
  For example with CsPad2x2, this will be a rank 3 array with 
  dimensions (185, 388, 2).

**********************
 User Function Output
**********************

The output of the function has three parts. 
First, a number of arrays to hold different parts of 
correlation analysis. These are typically terms in the calculation that a user may want
to save before forming the final answer, or that need to be collected at the viewer before
some part of the final answer can be formed. For instance with 
:ref:`g2`, there are three terms, one for the numerator, and two for the denominator.
::

  D x DetectorNDArray1   (array datatype=float64)
  D x DetectorNDArray2   (array datatype=float64)
  ...
  D x DetectorNDArrayN   (array datatype=float64)

Here D represents a smaller number of outputs. 
For example, whereas T may be a counter for the last 50,000 events, 
D may be a set of delays on a log scale, for example::

  D=[1,10,100,1000,10000].  

D is assumed to be in the same units as T, a 120hz counter.

The second part is counts for each of the delays::

  D x 1  delay counts    (array of int64)
                        
For this output counts, `counts[d]` is the number of pairs of times 
in T that are have a delay of d between them. A counts entry can be zero 
for a delay that has not been seen yet in the data.

The third output for F is a general array::

  DetetectorNDArray1     (array datatype=int8)
  
which applications can use for a variety of purposes. The intended application is
to record saturated pixels, pixels whose calibrated value that when over a threshold 
at some point and should thus be excluded from the presented results.

**********************
 Parallelization
**********************

The parallelization is achieved by distributing the detector NDArray elements
among the workers. That is this framework is presently only for
functions **F** where the calculations across NDArray elements are independent
of one another. Each worker gets a fraction of the NDArray elements.

**********************
 Using the Framework
**********************

To use this framework, the user does the following,

  * tell the framework what dataset to process and what detector data to extract
  * provides a mask file identifying what part of the detector data to process
  * implement worker code to calculate F on subset of the detector data
  * tells the framework how many output arrays it is computing
  * implement viewer code to plot/save the results
  * launch an MPI job

We will go through these steps in the tutorial that follows.
