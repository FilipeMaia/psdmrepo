
.. _tutorial:

################
 Tutorial
################

**************************
 Get/Build the Software
**************************

To get started with the software, make a new release directory and add the ParCorAna
package to it. Presently ParCorAna is not part of the ana release. For example, do
these commands::

  newrel ana-current ParCorAnaRel
  cd ParCorAnaRel
  sit_setup

  # now get a kerberos ticket to get code from the repository
  kinit   

  # now identify the most recent tag of ParCorAna
  psvn tags ParCorAna

  # suppose the last tag in the Vnn-nn-nn series is V00-00-08, then do
  addpkg ParCorAna V00-00-08
  scons

.. _configfile:

**************************
 Make a Config File
**************************

One first has to make a configuration file. This is a python file
that defines two dictionaries::

  system_params
  user_params

system_params defines a number of keys that the framework uses. user_params 
is only for the user module, the framework simple passes user_params through to the user module.

The simplest way to do this is to copy the default config file from the ParCorAna package.
From the release directory ParCorAnaRel where you have added the ParCorAna package, do::

  cp data/ParCorAna/default_params.py myconfig.py

Below we go over the config file. See comments in the config file for additional details.

The config file starts by importing modules used later. In particular::

  import numpy as np
  import psana
  import ParCorAna

It then defines the system_params dictionary to fill out::

  system_params={}

DataSource
=============

Next one specifies the datasource psana will read. It is recommended that one do this by 
specifying the run and experiment in separate variables. Then one can reuse those variables 
when specifying the h5output file to save results to.
::

  run = 1437
  experiment = 'xpptut13'
  system_params['dataset'] = 'exp=%s:run=%d' % (experiment, run) 

When doing online monitoring against live data, specify the ffb directory for the data, and
live mode. For example::

  system_params['dataset'] = 'exp=%s:run=%d:live:dir=/reg/d/ffb/xpp/xpptut13' % (experiment, run) 

This will start processing xtc files as soon as they appear on the ffb - usually a few seconds behind the shot.
Further by running with 6 servers on psfehq (see below), one should be able to keep up with reading the data.

If the DAQ streams for the run are not numbered 0-n, and you are using several servers to keep up with the 
data, you will need to explicitly list the streams. For example, if you are collecting from three DAQ streams
numbered 1,2 and 5, and also recording xtcav and other IOC's in stream 80 and 81 (and you need the IOC data
when deciding what events to include in the G2 calculation) one would do::

  system_params['dataset'] = 'exp=%s:run=%d:live:stream=1,2,5,80,81:dir=/reg/d/ffb/xpp/xpptut13' % (experiment, run) 
Then set numservers (below) to 3 for the three DAQ streams, 1,2 and 5. Note, if you do not need the
IOC streams for the G2 calculation, do not include them, it will speed up processing.

These parameters::

  system_params['src']       = 'DetInfo(XppGon.0:Cspad.0)'
  system_params['psanaType'] = psana.CsPad.DataV2

tell the framework the source, and psana data type the framework needs to distribute to the workers.
The default values are for xpp tutorial data with full cspad. Because we did an "import psana" at the
top of the config file, we can directly use values like psana.CsPad.DataV2 in the config file.

Psana Configuration
====================

The parameters::

  system_params['psanaOptions']          # dictionary of psana options
  system_params['ndarrayProducerOutKey'] # event key string for ndarray producer output
  system_params['ndarrayCalibOutKey']    # event key string for NDarrayCalib output
  system_params['outputArrayType']       # such as psana.ndarray_float64_3

do the following

 * specify a psana configuration that loads the appropriate ndarray producer psana
   module for the detector data, and optionally (but most always) the NDarrayCalib psana module 
   that calibrates the ndarray
 * specify output keys for the two modules
 * specify the final array type the psana modules output

The framework specifies the psana configuration through a python dictionary rather than a config file.
To simplify the setting of these parameters, default_params.py uses a utilty function makePsanaOptions.
This function automatically figures out the correct psana modules to load and their options based on
the detector type.

Generally users will not need to modify the code in default_params.py which reads::

  system_params['ndarrayProducerOutKey'] = 'ndarray'
  system_params['ndarrayCalibOutKey'] = 'calibrated'    # set to None to skip calibration

  system_params['psanaOptions'], \
  system_params['outputArrayType'] = ParCorAna.makePsanaOptions(
                                       srcString=system_params['src'],
                                       psanaType=system_params['psanaType'],
                                       ndarrayOutKey=system_params['ndarrayProducerOutKey'],
                                       ndarrayCalibOutKey=system_params['ndarrayCalibOutKey']
                                     )

However users may want to adjust options to the calibration modules. For example, to add gain, one can add the
following line after the above::

  system_params['psanaOptions']['ImgAlgos.NDArrCalib.do_gain'] = True

default_params.py includes code that allows one to do::

  python default_params.py

to make sure there are no errors in the file, as well as to pretty print the final system_params and
user_params dictionaries. The resulting 'psanaOptions' from the above call to makePsanaOptions are::

  'psanaOptions': {'CSPadPixCoords.CSPadNDArrProducer.is_fullsize': 'True',
                   'CSPadPixCoords.CSPadNDArrProducer.key_out': 'ndarray',
                   'CSPadPixCoords.CSPadNDArrProducer.outkey': 'ndarray',
                   'CSPadPixCoords.CSPadNDArrProducer.outtype': 'double',
                   'CSPadPixCoords.CSPadNDArrProducer.source': 'DetInfo(XppGon.0:Cspad.0)',
                   'ImgAlgos.NDArrCalib.below_thre_value': 0,
                   'ImgAlgos.NDArrCalib.do_bkgd': False,
                   'ImgAlgos.NDArrCalib.do_cmod': True,
                   'ImgAlgos.NDArrCalib.do_gain': False,
                   'ImgAlgos.NDArrCalib.do_mask': False,
                   'ImgAlgos.NDArrCalib.do_nrms': False,
                   'ImgAlgos.NDArrCalib.do_peds': True,
                   'ImgAlgos.NDArrCalib.do_stat': True,
                   'ImgAlgos.NDArrCalib.do_thre': False,
                   'ImgAlgos.NDArrCalib.fname_bkgd': '',
                   'ImgAlgos.NDArrCalib.fname_mask': '',
                   'ImgAlgos.NDArrCalib.key_in': 'ndarray',
                   'ImgAlgos.NDArrCalib.key_out': 'calibrated',
                   'ImgAlgos.NDArrCalib.masked_value': 0,
                   'ImgAlgos.NDArrCalib.source': 'DetInfo(XppGon.0:Cspad.0)',
                   'ImgAlgos.NDArrCalib.threshold': 0,
                   'ImgAlgos.NDArrCalib.threshold_nrms': 3,
                   'modules': 'CSPadPixCoords.CSPadNDArrProducer ImgAlgos.NDArrCalib'}


Worker Storage
================

The psana calibration module NDArrCalib defaults to creating ndarrays of double. 
These are 8 bytes wide. Each worker stores a portion of this ndarray. To guarantee no 
loss of precision, workers should store results in the same data format - i.e, float64.
However for large detectors and long correlation types, this may require too much 
memory. For full cspad where all pixels are included in the mask, and 50,000 times are stored
on the workers, this amounts to 50,000*(32*388*185)*8=855GB of memory that must be 
distributed amoung all the workers. If each host has 24GB, one would 
have to use 36 hosts. If each host runs 12 MPI ranks, we need 432 ranks for the workers.

A simple way to use less memory, is to have the workers store the detector data as 4
byte floats. This is what is done in default_params.py::

  system_params['workerStoreDtype'] = np.float32


Mask File
===========

You need to provide the framework with a mask file for the detector data. This is a 
numpy array with the same dimensions as the ndarray that the psana ndarray producer 
module creates. This is not necessarily a 2D image that is easy to plot. In addition, 
you should create a testing mask file that masks a very small number of pixels 
(10 to 100). The small number of pixels in the test mask file allows one to run 
a simple alternative calculation against the data to validate the calculation done
through the framework.
::

  system_params['maskNdarrayCoords'] = 'maskfile.npy' # not created yet
  system_params['testMaskNdarrayCoords'] = 'testmaskfile.npy' # not created yet


Number of Servers
===================

The servers are responsible for working through the data, breaking up an ndarray of detector 
data, and scattering it to the workers. When developing, we usuaully specify 
one server. When analyzing data in live mode, we usually specify 6 servers, or however many
DAQ streams there are in the run. The framework sets things up so that each server only processes
one stream. As long as each server can run at 20hz it will keep up with live 120hz data. 
If you are analyzing xtcav data, then each server will process 2 or more streams. The framework 
outputs timing at the end which gives us an idea of how fast or slow the servers are.
Specifying more than 6 servers will not help, rather it will waste too many ranks on servers.

In index mode, specifying more than six servers can help the servers run faster. However usually
the bottleneck will be with the workers, and more than six servers is not neccessary. The framework
outputs timing information at the end of runs that allow one to see what part of the system
is slow.

By default, the framework will pick distinct hosts to run the servers on. Distributing the I/O
among several hosts seems to improve performance, but this is debatable.
::

  system_params['numServers'] = 1
  system_params['serverHosts'] = None     # system selects which hosts to use

Times, Delays, update
========================
::

  system_params['times'] = 50000
  system_params['delays'] = ParCorAna.makeDelayList(start=1,
                                                    stop=25000, 
                                                    num=100, 
                                                    spacing='log',  # can also be 'lin'
                                                    logbase=np.e)
  system_params['update'] = 0      # how frequently to update, units are events

These parameters specify how many events we will store, and what the delays are. 
If one stores 50,000 events but there are 100,000 events in the dataset, the 
framework will start overwriting the oldest data at event 50,001. 

Above we are specifying 100 delays that are logarithmically spaced from 1 to 25,000 bu
using a utility function in ParCorAna. However one can set their own delays::

  system_params['delays'] =  [    1,    10, 100, 1000]

Periodically, the workers are told to calculate correlation for their pixels. The framework
gathers these results from all the workers and sends it to the viewer. When 'update' is 0, 
this just happens once at the end. Otherwise 'update' specifies the number of events between
these gathers. If one is analyzing live data and producing plots, one could specify 360 to get a 
plot every 3 seconds - however gathering results at the viewer can be expensive, and 3 seconds may
be too frequent to keep up with the data (depending on the problem size).

User Module
========================
::

  import ParCorAna.UserG2 as UserG2
  system_params['userClass'] = UserG2.G2atEnd

The userClass is where users hook in their worker code. We will be using the example 
class in the ParCorAna package - G2atEnd does a simplified version of the G2 
calculation used in XCS - however the file UserG2.py goes over three ways to do the G2
calculation:

 * **G2atEnd** workers store data during each event, do a O(T*D) calculation during updates (where T is number of times, and D is number of delays)
 * **G2IncrementalAccumulator** workers do O(D) work with each event, doing correlation over all times
 * **G2IncrementalWindowed** workers do O(D) work with each event, doing a windowed correlation, over the last T times

More on this in section XXX???

H5Output
=============
The system will optionally manage an h5output file. This is not a file for collective MPI
writes. Within the user code, only the viewer rank should write to the file. The viewer
will receive an open group to the file at run time. 

Set h5output to None if you do not want h5 output - important to speed up online monitoring with 
plotting.

The system will recognize %T in the filename and replaces it with the current time in the format
yyyymmddhhmmss. (year, month, day, hour, minute, second). It will also recognize %C for a three
digit one up counter. When %C is used, it looks for all matching files on disk, selects the
one with the maximum counter value, and adds 1 to that for the h5output filename.

Testing is built into the framework by allowing one to run an alternative calculation
that receives the same filtered and processed events at the main calculation. When the
alternative calcuation is run, the framework uses the testh5output argument for the
filename.
::

  system_params['h5output'] = 'g2calc_%s-r%4.4d.h5' % (experiment, run)
  system_params['testh5output'] = 'g2calc_test_%s-r%4.4d.h5' % (experiment, run)


example of using %T and %C, note the %% in the value to get one % in the string after 
expanding experiment and run::

  system_params['h5output'] = 'g2calc_%s-r%4.4d_%%T.h5' % (experiment, run)
  system_params['h5output'] = 'g2calc_%s-r%4.4d_%%C.h5' % (experiment, run)

For re-running the analysis, set the below to True to overwrite existing h5 files::

  system_params['overwrite'] = False   

While the analysis is running, it adds the extension .inprogress to the output file.
The framework will never overwrite a .inprogress file, even if 'overwrite' is True.
If analysis crashed due to an error, these leftover files need to be manually removed.

Debugging/Develepment Switches
=====================================
::

  system_params['verbosity'] = 'INFO'
  system_params['numEvents'] = 0
  system_params['testNumEvents'] = 100

These options are useful during development or debugging. Setting the verbosity to
DEBUG greatly increases the amount of output. It can trigger additional runtime checks.
Typically it is only the first worker that outputs a message, as all the workers do the same 
thing.

One can also limit the number of events processes, and specify the number of event to process
during testing (for both the main code, and the alternative calculation). 


User Parameters
====================
The user_params dictionary is where users put all the options that the G2 calculation
will use.


Color/Bin/Label File
----------------------
This is a parameter that the UserG2 needs - a file that labels the detector pixels
and determines which pixels are averaged together for the delay curve. It bins the pixels
into groups. In this package, we call this a 'color' file following conventions in MPI
for grouping different ranks. More on this in the next section::

  user_params['colorNdarrayCoords'] = 'colorfile.npy' # not created yet

Fine Color/Bin/Label File
----------------------
This is a another parameter that the UserG2 needs. It is a color file that is used to 
replace classes of pixels with their average value. This is applied to the IP and IF matricies
before forming the final G2 curves. Note, these modified IP and IF matrices are used for calculating
the G2 delay curves, the modified IP/IF are not saved to the hdf5 file::

  user_params['colorFineNdarrayCoords'] = 'filnColorfile.npy' # not created yet

Filtering Parameters
-----------------------
Often G2 calculations need to adjust/filter the data. The UserG2 module sets several 
parameters that it makes use of::

  user_params['saturatedValue'] = (1<<15)
  user_params['LLD'] = 1E-9
  user_params['notzero'] = 1E-5

Plotting
----------
The UserG2 module provides an example of how to use the psmon package to plot.
This is the preffered method to plot for online monitoring where the analysis job
runs on a batch farm. For now we set this value to False. Using psmon for plotting
will be covered in section XXX???
::

  user_params['psmon_plot'] = False


***************************
 Create a Mask/Color File
***************************
The system requires a mask file that identifies the pixels to process.
Reducing the number of pixels processed can be the key to fast feedback during an experiment.
A userClass for correlation analysis will typically use two 'color' files to label
pixels to average together. The first, a coarser one, is used to average pixels for the
final delay curves. The second, a finer one, is used to replace classes of pixels with their
average value before forming the delay curves. In addition to the mask file for analyzing data, 
users should produce a testmask file for testing their compution. 
This file should only compute on a few (10-100) pixels.


The ParCorAna package provides a tool to make mask and color files in the numpy ndarray
format required. To read the tools help do::

  parCorAnaMaskColorTool -h

(Another tool to look at is roicon, also part of the analysis release). The command::

  parCorAnaMaskColorTool --start -d 'exp=xpptut13:run=1437' -t psana.CsPad.DataV2 -s 'DetInfo(XppGon.0:Cspad.0)' -n 300 -c 6 --finecolor 18

Will produce a mask, testmask, color and fineColor file suitable for this tutorial::

  xpptut13-r1437_XppGon_0_Cspad_0_mask_ndarrCoords.npy  
  xpptut13-r1437_XppGon_0_Cspad_0_testmask_ndarrCoords.npy  
  xpptut13-r1437_XppGon_0_Cspad_0_color_ndarrCoords.npy 
  xpptut13-r1437_XppGon_0_Cspad_0_finecolor_ndarrCoords.npy 

Note that our input will be ndarr files, not image files. The mask file is only  0 or 1. It is 1
for pixels that we **INCLUDE**. The color file uses 6 colors (since we gave the -c 6 option to the tool. 
The finecolor file will have 18 colors (since we used the --finecolor 18 option).
As an example, these colors bin pixels based on intensity. In practice users will want to bin pixels
based on other criteria.

The tool parCorAnaMaskColorTool can produce a great deal of output that can be ignored.
To deal with converting from images to ndarrays, it is neccessary to work with geometry files.
If a geometry file is not present for your experiment, one should be deployed into the calibration
area. One can also specify a geometry file with the -g file. 

Often people will edit image files to produce the mask and color file. These need to then be converted
back to ndarrays. The help for parCorAnaMaskColorTool explains how to do this. One issue with this though,
is that sometimes the geometry files map two ndarray pixels to the same image pixel, and do not map some
of the ndarray pixels to any image pixel. This means that the ndarray mask or color file produced from the
image file will have a few imperfections. For the cspad in the xpp tutorial data, the number of such pixels 
is quite small. The tool reports on this pixels. It also reports on the location of the 10 pixels chosen
for the mask.

Once you have modified these files, or produced similarly formatted files you are ready for the 
next step.

Add to Config
==================

Now in myconfig.py, set the mask, testmask, and color file::

  system_params['maskNdarrayCoords'] = 'xpptut13-r1437_XppGon_0_Cspad_0_mask_ndarrCoords.npy'
  system_params['testMaskNdarrayCoords'] = 'xpptut13-r1437_XppGon_0_Cspad_0_testmask_ndarrCoords.npy'
  user_params['colorNdarrayCoords'] = 'xpptut13-r1437_XppGon_0_Cspad_0_color_ndarrCoords.npy'

Note that the last parameter is to the user_params - the framework knows nothing about the coloring.

********************
Check Config File
********************

Once you have modified the config file, it is a good idea to check that it runs as python code, i.e, that
all the imports work and the syntax is correct::

  python myconfig.py

The config file does a pretty-print of the two dictionaries defined.

.. _runlocal:

***********************************
Run Software 
***********************************

Now you are ready to run the software. To test using a few cores on your local machine, do::

  mpiexec -n 4 parCorAnaDriver -c myconfig.py -n 100

This should run without error. Even though we are only running on 100 events, the viewer will be
gathering 100 (delays) * 32 * 388 * 185 (cspad dimensions) * 8 (float64) = 1,837,568,000 bytes, 
or close to 2GB.

***********************************
Results
***********************************
You can get a listing of what is in the output file by doing::

  h5ls -r g2calc_xpptut13-r1437.h5

The h5 file contains two groups at the root level::

  /system
  /user

In /system, one finds::

  /system/system_params    Dataset 
  /system/user_params      Dataset
  /system/color_ndarrayCoords Dataset
  /system/mask_ndarrayCoords Dataset 

The first two are the system_params and user_params dictionaries from the config file.

The latter two are the mask and color ndarrays specified in the system_params.

In /user one finds whatever the user viewer code decides to write. The example 
UserG2 module writes, for example::

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

*******************
Plotting
*******************
To do plots, set the following in the config file::

  user_params['psmon_plot'] = True

When plotting, you may not want to produce the h5output as well. If so, also set::

  system_params['h5output'] = None

When running with the psmon_plot parameter to True, In the output, one should see outupt similar to::

  2015-05-08 14:39:16,214 - viewer-rnk:2 - INFO - Run cmd: psplot --logx -s psana1501 -p 12301 MULTI

The command::

  psplot --logx -s psana1501 -p 12301 MULTI

or something similar is what one runs to see the plots. The host (psana1501 in above) will be 
different when you run. It is the host that the viewer is on. The port can be changed by setting
the option::

  user_params['psmon_port'] = 12301

in the config file.

You may not want to use the --logx option if the delays are linearly spaced. If you use the 
--logx option and get an error, it is a recent option that may not be in the current analysis
release yet. Do::

  newrel ana-current myrel
  cd myrel
  sit_setup
  addpkg psmon HEAD
  scons

and try the command again.

.. _runonbatch:

*****************************
Running on the Batch System
*****************************
When running on the batch system, for example online monitoring in xcs, one would do something like::

  bsub -q psfehpriorq -I -n 150 parCorAnaDriver -c myconfig.py

The -I option means interactive, so that the program output will go to the screen. This will let
you see the psplot command. However all you need to know is what host the viewer is on, and this is
typically the first host. Doing::
  
 bjobs -w

Will show you what the first host is as well.

*****************************
Timing
*****************************
To see if you can keep up with live data, look at the output messages. You will see lines like

TODO

*****************************
Testing
*****************************
see the testing page

*****************************
UserG2
*****************************
see the :ref:`usercode` section of the :ref:`framework` page.

..  LocalWords:  ParCorAna ParCorAnaRel cd kerboses
