
.. _architecture:

#########################
 Software Architecture
#########################

Below we take a top down approach to summarizing the components of the framework.

**************************
mpiFnDriver
**************************

This is where the system starts. Use the -h option to get help on running it. A path it is is already setup in 
an analysis release with the ParCorAna package. Key steps that it takes:

* reads the config file. Some config file options can be overriden with command line options, namely:

  ** verbosity - more for debugging
  ** numevents - more for debugging, end early
  ** h5output  - override output file
  ** elementsperworker  - more for debugging

* Instantiates a framework and runs it:

  ** framework = CommSystemFramework(system_params, user_params)
  ** framework.run()

**************************
CommSystemFramework
**************************

this is what the mpiFnDriver kicks off. This handles the mpi
communication between master/servers/workers/viewer. It is meant to be 
agnostic of the kind of calculation being done. 

Key steps:

* identifyServerRanks: looks at numservers, identifies server ranks.
* identifyCommSubsystems: splits ranks into servers, workers, viewer, master. 
  Creates intra-communicators for collective communication between

  ** viewer <-> workers
  ** each server <-> workers

* loads mask file
* Creates XCorrBase - part of the framework. The CommSystem talks to the XCorrBase.
  The XCorrBase talks to the user module. 
  XCorrBase knows about the kind of function F being implemented. 
  It handles 

** splitting ndarrays among the workers, sending them flattened 1D arrays of elements
** gathering results
** delivering and gathered results ad reassembled NDArrays to viewer code

**************************
runCommSystem
**************************
this does different things depending on whether or not the rank is
a server, worker, viewer, or the master. We describe this below.

===========
Server
===========

===========
Worker
===========

===========
Viewer
===========

===========
Master
===========

