.. _tips:

##########
Tips
##########

Below we go over tips for running the software. The goal is to be able to run it in live mode for the next XCS experiment using the G2 calculation.

*****************************
Setup Release
*****************************
make sure it is based on ana-current, or at least ana-0.15.6 to get the fast infiniband MPI.

* Do addpkg ParCorAna V00-00-26 (or whatever is the latest version)
* scons
* sit_setup (to be able to run software)

*****************************
Assembing files
*****************************

* deploy a geometry file
* take a run with cspad
* run the parCorAnaMaskColorTool to make a sample full mask, sample color, sample finecolor in both image and ndarray coords, as well as iX, iY files for image <-> ndarray conversion
* prepare the real mask, color and finecolor file as ndarrays, i.e, (32, 185, 388) sized files. Fix up in ipython if need by. All should be int matricies, mask could be np.int8, the others np.int32.
* mask=1 means pixel is included.
* pixels with 0 in the color and finecolor files are not processed.
* Make sure where color > 0 finecolor > 0 as well.
* no negative values in mask, color, finecolor.
* if many pixels in color==0, consider making them 0 in mask to speed things up

**********************
Config File
**********************
Below is a sample config file. This is based on what is checked into ParCorAna at data/config_fullcspad_accum_ffb_plot.py. Below we go over what to change
::

  import numpy as np
  import psana
  import ParCorAna
  
  system_params={}
  experiment = 'xcs84213'; run=117 
  system_params['dataset'] = 'exp=%s:run=%d:live:stream=0-5:dir=/reg/d/ffb/xcs/%s/xtc' % (experiment, run, experiment) 
  system_params['src']       = 'DetInfo(XcsEndstation.0:Cspad.0)'
  system_params['psanaType'] = psana.CsPad.DataV2 
  
  system_params['ndarrayProducerOutKey'] = 'ndarray'
  system_params['ndarrayCalibOutKey'] = 'calibrated' 
  
  system_params['psanaOptions'], \
      system_params['outputArrayType'] = ParCorAna.makePsanaOptions(
                                           srcString=system_params['src'],
                                           psanaType=system_params['psanaType'],
                                           ndarrayOutKey=system_params['ndarrayProducerOutKey'],
                                           ndarrayCalibOutKey=system_params['ndarrayCalibOutKey']
                                         )
  
  system_params['workerStoreDtype'] = np.float32 
  
  system_params['maskNdarrayCoords'] = 'mask.npy'         # CHANGE THIS
  system_params['testMaskNdarrayCoords'] = 'testmask.npy' # CHANGE THIS
  
  system_params['numServers'] = 6
  system_params['serverHosts'] = None
  
  system_params['times'] = 35000     # number of distinct times that each worker holds onto
  
  eventsPerSecond = 120
  numSeconds = 30
  system_params['update'] = numSeconds*eventsPerSecond
  
  
  ######### delays ############
  system_params['delays'] = ParCorAna.makeDelayList(start=1,
                                                    stop=34500, 
                                                    num=100,
                                                    spacing='log',  # can also be 'lin'
                                                    logbase=10.0)
  
  ######## User Module ########
  import ParCorAna.UserG2 as UserG2
  system_params['userClass'] = UserG2.G2IncrementalAccumulator
  
  system_params['h5output'] = None # 'xxx.h5' # None # 'g2calc_%s-r%4.4d_%%C.h5' % (experiment, run)
  system_params['testH5output'] = 'g2calc_test_%s-r%4.4d.h5' % (experiment, run)
  
  
  # example of using %T and %C, note the %% in the value to get one % in the string after 
  # expanding experiment and run:
  
  # system_params['h5output'] = 'g2calc_%s-r%4.4d_%%T.h5' % (experiment, run)
  # system_params['h5output'] = 'g2calc_%s-r%4.4d_%%C.h5' % (experiment, run)
  
  ## overwrite can also be specified on the command line, --overwrite=True which overrides what is below
  system_params['overwrite'] = True   # if you want to overwrite an h5output file that already exists
  
  ######## verbosity #########
  # verbosity can be one of INFO, DEBUG, WARNING (levels from the Python logging module)
  system_params['verbosity'] = 'INFO'
  
  ######## numevents #########
  # numevents - 0 or None means all events. This is primarily a debugging/development switch that
  # can be overriden on the command line
  system_params['numEvents'] = 0
  system_params['testNumEvents'] = 100
  
  ##################################################
  ############ USER MODULE - G2 CONFIG #############
  user_params = {}
  
  # the partition is a numpy array of int's. 0 and negative int's are ignored. int's that are positive
  # partition the elements. That is all elements with '1' form one delay curve, likewise all elements that are '2'
  # form another delay curve.
  user_params['colorNdarrayCoords'] = 'L842_mask_dynamic_ndarray.npy'
  user_params['colorFineNdarrayCoords'] ='L842_mask_static_ndarray.npy'
  user_params['saturatedValue'] = (1<<15)
  user_params['LLD'] = 1E-9
  user_params['notzero'] = 1E-5
  user_params['psmon_plot'] = True 
  user_params['plot_colors'] = [1,4,6,8]
  user_params['print_delay_curves'] = False
  
  user_params['debug_plot'] = False
  user_params['iX'] = None
  user_params['iY'] = None
  
  # to set a different port for psmon plotting, change this
  # user_params['psmon_port'] = 12301
  user_params['ipimb_threshold_lower'] = .05
  user_params['ipimb_srcs'] = []
  
  ##################
  # for debugging this params file, run it as a python script. It will
  # print the content of the two dictionaries.
  
  if __name__ == '__main__':
      print "######## system_params dict #########"
      from pprint import pprint
      pprint(system_params)
      print "######## user_params dict #########"
      from pprint import pprint
      pprint(user_params)
      ParCorAna.checkParams(system_params, user_params, checkUserParams=True)

Things to change/note with the above config file

* change experiment/run for current. Operationally, instead of changing the config file for each run, it may be easier to use the -d command line option to the parCorAnaDriver to specify the new dataset, with the new run.
* specify live mode and the ffb directory for the data for online monitoring.
* explicitly list the 6 daq streams in the dataset string.
* plug in your mask, testmask, color, and finecolor files
* use 6 servers (or as many DAQ streams as you have)
* choose how many times to store: 35,000, 50,000?
* update - choose plot frequency in events (it is not real time)
* choose delays
* Use UserG2.G2IncrementalAccumulator to get fastest processing for online monitoring
* set h5output to None. If specifying h5output - you may get better performance writing to ana filesystem (i.e, ftc directory of experiment, etc) rather then home directory, however you will be computing on the psfehq. psfehq has fast access to the data on ffb for reading, but not fastest access to ana for writing.
* consider adjusting the 'saturatedValue' from 1<<15 to something more reasonable. 1<<15 makes sense for raw data, but this is applied to calibrated data. Decide what value for calibrated pixels you want to use to exclude pixels from delay curve calculations (if any, probably little harm to leave this too high to be effective).
* The 'LLD' parameter is not used, modify the UserG2.py if this is important ('notzero' is used to replace negative and small values with that value).
* Choose some plot colors to plot
* You will probably not need to use 'debug_plot' which is what uses iX and iY.
* Set the ipimb parameters, modify UserG2.py server callbacks, if required.

***************************
Check your config file
***************************
Run the config file through python, for example::

  python myconfig.py

It can quickly identify some simple errors, save time.

***************************
Launch Jobs
***************************
For online on the priority queue, first clear the queue of unwanted jobs. These commands
could be useful to see what it going on::

  bqueues | grep "PEN\|ps"
  bjobs -q psfehprioq -u all
  bjobs -q psfehq -u all

Jobs launched on psfehprioq should pre-empt those on psfehq.

Decide how many cores to run on. Look at the hosts for psfehprior by looking at psfehfarm::
 
  bhosts -w psfehfarm

Sum up the MAX column, don't count unavail, I think also don't count closed_Adm, but not sure. 
When I recently did it, I saw 14 hosts that could run 16 jobs each - meaning a max of 224 ranks
for the job.

Next launch the job. It is very useful to see output as it scrolls by. I typically run in 'interactive mode'
::

  bsub -a mympi -n 200 -q psfehprioq -I parCorAnaDriver -c myconfig.py

Keep an eye out for where the job is starting to identify the viewer host.
You can also wait for the line that outputs the psplot command. Then run that command to 
see the plots, it will be something like::

  psplot --logx -s psana1620 -p 12301 MULTI  

but you may have to replace psana1620 with your host.


*********************
Debugging/Problems
*********************
Debugging an MPI program can be difficult. I usually do printf. Not the logger member to UserG2,
along with the logInfo function. To debug most of the callbacks, you can do the alt_test. I.e::

  parCorAnaDriver -c myconfig.py --test_alt

This runs outside MPI. You can set a breakpoint in many UserG2 callbacks, just not the worker
ones for before/after data insert, and workerCalc.

Also consider running with -v debug to get debugging output. You can try setting 'debug_plot' to 
get the debugging plot.

***********************
Understanding Output
***********************
Below is the output of a run I did, some comments are below.
::

  psana1501:~/rel/ParCorAnaFullRun/sikorski_files $ bsub -q psfehq -n 150 -a mympi -I parCorAnaDriver -c config_fullcspad_accum_ffb.py 
  Warning: job being submitted without an AFS token.
  Job <446125> is submitted to queue <psfehq>.
  <<Waiting for dispatch ...>>
  <<Starting on psana1620.pcdsn>>
  parCorAnaDriver rank=0 before first Collective MPI call (MPI_Barrier). If no output follows, there is a problem with the cluster.
  parCorAnaDriver rank=0 after first collective MPI call. Elapsed time: 12.91 sec
  2015-06-28 22:54:47,324 - master-rnk:1 - INFO - G2IncrementalAccumulator: object initialized
  2015-06-28 22:54:47,329 - master-rnk:1 - INFO - server host assignment:rnk=16->host=['psana1602'], rnk=32->host=['psana1603'], rnk=48->host=['psana1604'], rnk=64->host=['psana1605'], rnk=80->host=['psana1608'], rnk=96->host=['psana1610']
  2015-06-28 22:54:47,335 - viewer-rnk:0 - INFO - G2IncrementalAccumulator: object initialized
  2015-06-28 22:54:47,338 - worker-rnk:2 - INFO - G2IncrementalAccumulator: object initialized
  2015-06-28 22:54:47,424 - viewer-rnk:0 - INFO - UserG2.viewerInit: mask included pixels contain colors: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] with counts: [1282, 2154, 2776, 3369, 3999, 4600, 5155, 5810, 6353, 6910, 7599, 8130, 8741, 9326, 9937, 10072, 8761, 7492, 3270, 705]
  2015-06-28 22:54:47,425 - viewer-rnk:0 - INFO - UserG2.viewerInit: mask included pixels contain finecolors: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110] with counts: [95, 195, 244, 279, 292, 348, 350, 384, 363, 448, 438, 460, 454, 519, 522, 543, 525, 610, 588, 634, 619, 671, 673, 714, 724, 732, 764, 788, 795, 814, 845, 876, 874, 887, 914, 962, 948, 964, 983, 1036, 1045, 1067, 1046, 1113, 1108, 1149, 1128, 1200, 1184, 1203, 1204, 1264, 1280, 1293, 1250, 1360, 1347, 1387, 1371, 1392, 1445, 1466, 1454, 1467, 1542, 1498, 1564, 1541, 1590, 1626, 1606, 1599, 1677, 1684, 1752, 1682, 1746, 1744, 1843, 1762, 1817, 1836, 1886, 1867, 1849, 1857, 1856, 1692, 1682, 1640, 1628, 1594, 1515, 1432, 1463, 1449, 1377, 1287, 1186, 885, 712, 573, 503, 426, 304, 214, 166, 131, 46, 15]
  2015-06-28 22:54:47,425 - viewer-rnk:0 - INFO - Initialized psmon. viewer host is: psana1620.pcdsn
  2015-06-28 22:54:47,425 - viewer-rnk:0 - INFO - *********** PSPLOT CMD *************
  2015-06-28 22:54:47,425 - viewer-rnk:0 - INFO - Run cmd: psplot --logx -s psana1620.pcdsn -p 12301 MULTI
  2015-06-28 22:54:47,425 - viewer-rnk:0 - INFO - *********** END PSPLOT CMD *************
  2015-06-28 22:54:47,431 - server-rnk:80 - INFO - G2IncrementalAccumulator: object initialized
  2015-06-28 22:54:47,505 - server-rnk:64 - INFO - G2IncrementalAccumulator: object initialized
  2015-06-28 22:54:47,528 - server-rnk:48 - INFO - G2IncrementalAccumulator: object initialized
  2015-06-28 22:54:47,702 - server-rnk:32 - INFO - G2IncrementalAccumulator: object initialized
  2015-06-28 22:54:47,961 - server-rnk:16 - INFO - G2IncrementalAccumulator: object initialized
  2015-06-28 22:54:48,749 - server-rnk:96 - INFO - G2IncrementalAccumulator: object initialized
  2015-06-28 22:54:49.142 [WRN] {NDArrIOV1} NDArrIOV1.cpp:306 - NDArray file:
    /reg/d/psdm/XCS/xcs84213/calib/CsPad::CalibV1/XcsEndstation.0:Cspad.0/common_mode/0-end.data
    does not have enough data: read 3 numbers, expecting 4
  2015-06-28 22:54:49.238 [WRN] {NDArrIOV1} NDArrIOV1.cpp:306 - NDArray file:
    /reg/d/psdm/XCS/xcs84213/calib/CsPad::CalibV1/XcsEndstation.0:Cspad.0/common_mode/0-end.data
    does not have enough data: read 3 numbers, expecting 4
  2015-06-28 22:54:49.266 [WRN] {NDArrIOV1} NDArrIOV1.cpp:306 - NDArray file:
    /reg/d/psdm/XCS/xcs84213/calib/CsPad::CalibV1/XcsEndstation.0:Cspad.0/common_mode/0-end.data
    does not have enough data: read 3 numbers, expecting 4
  2015-06-28 22:54:49.312 [WRN] {NDArrIOV1} NDArrIOV1.cpp:306 - NDArray file:
    /reg/d/psdm/XCS/xcs84213/calib/CsPad::CalibV1/XcsEndstation.0:Cspad.0/common_mode/0-end.data
    does not have enough data: read 3 numbers, expecting 4
  2015-06-28 22:54:49.423 [WRN] {NDArrIOV1} NDArrIOV1.cpp:306 - NDArray file:
    /reg/d/psdm/XCS/xcs84213/calib/CsPad::CalibV1/XcsEndstation.0:Cspad.0/common_mode/0-end.data
    does not have enough data: read 3 numbers, expecting 4
  2015-06-28 22:54:50.525 [WRN] {NDArrIOV1} NDArrIOV1.cpp:306 - NDArray file:
    /reg/d/psdm/XCS/xcs84213/calib/CsPad::CalibV1/XcsEndstation.0:Cspad.0/common_mode/0-end.data
    does not have enough data: read 3 numbers, expecting 4
  2015-06-28 22:55:33,722 - master-rnk:1 - INFO - Current data rate is 25.89 Hz. 1201 events processed
  2015-06-28 22:55:51,755 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=2376 took 0.0000 sec
  2015-06-28 22:55:53,212 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 1.457 sec
  2015-06-28 22:55:53,212 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 1.471 sec
  2015-06-28 22:55:53,579 - master-rnk:1 - INFO - Current data rate is 60.48 Hz. 2402 events processed
  2015-06-28 22:55:56,602 - viewer-rnk:0 - INFO - viewerFormNDarrays took 3.390 sec
  2015-06-28 22:56:12,119 - master-rnk:1 - INFO - Current data rate is 64.78 Hz. 3603 events processed
  2015-06-28 22:56:28,648 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=4753 took 0.0000 sec
  2015-06-28 22:56:29,548 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.901 sec
  2015-06-28 22:56:29,548 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.913 sec
  2015-06-28 22:56:30,235 - master-rnk:1 - INFO - Current data rate is 66.29 Hz. 4804 events processed
  2015-06-28 22:56:32,255 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.707 sec
  2015-06-28 22:56:47,022 - master-rnk:1 - INFO - Current data rate is 71.54 Hz. 6005 events processed
  2015-06-28 22:57:03,038 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=7130 took 0.0000 sec
  2015-06-28 22:57:03,944 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.906 sec
  2015-06-28 22:57:03,944 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.919 sec
  2015-06-28 22:57:05,046 - master-rnk:1 - INFO - Current data rate is 66.63 Hz. 7206 events processed
  2015-06-28 22:57:06,629 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.685 sec
  2015-06-28 22:57:22,187 - master-rnk:1 - INFO - Current data rate is 70.07 Hz. 8407 events processed
  2015-06-28 22:57:38,197 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=9507 took 0.0000 sec
  2015-06-28 22:57:39,107 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.924 sec
  2015-06-28 22:57:39,107 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.910 sec
  2015-06-28 22:57:40,552 - master-rnk:1 - INFO - Current data rate is 65.40 Hz. 9608 events processed
  2015-06-28 22:57:41,888 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.781 sec
  2015-06-28 22:57:58,078 - master-rnk:1 - INFO - Current data rate is 68.53 Hz. 10809 events processed
  2015-06-28 22:58:13,977 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=11884 took 0.0000 sec
  2015-06-28 22:58:14,890 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.913 sec
  2015-06-28 22:58:14,890 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.928 sec
  2015-06-28 22:58:16,678 - master-rnk:1 - INFO - Current data rate is 64.57 Hz. 12010 events processed
  2015-06-28 22:58:17,612 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.722 sec
  2015-06-28 22:58:34,376 - master-rnk:1 - INFO - Current data rate is 67.86 Hz. 13211 events processed
  2015-06-28 22:58:49,982 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=14261 took 0.0000 sec
  2015-06-28 22:58:50,897 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.915 sec
  2015-06-28 22:58:50,897 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.929 sec
  2015-06-28 22:58:53,138 - master-rnk:1 - INFO - Current data rate is 64.01 Hz. 14412 events processed
  2015-06-28 22:58:53,611 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.714 sec
  2015-06-28 22:59:11,070 - master-rnk:1 - INFO - Current data rate is 66.97 Hz. 15613 events processed
  2015-06-28 22:59:26,321 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=16638 took 0.0000 sec
  2015-06-28 22:59:27,241 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.920 sec
  2015-06-28 22:59:27,241 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.934 sec
  2015-06-28 22:59:29,955 - master-rnk:1 - INFO - Current data rate is 63.60 Hz. 16814 events processed
  2015-06-28 22:59:30,068 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.827 sec
  2015-06-28 22:59:48,031 - master-rnk:1 - INFO - Current data rate is 66.44 Hz. 18015 events processed
  2015-06-28 23:00:02,939 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=19015 took 0.0000 sec
  2015-06-28 23:00:03,850 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.911 sec
  2015-06-28 23:00:03,850 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.925 sec
  2015-06-28 23:00:06,600 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.750 sec
  2015-06-28 23:00:06,792 - master-rnk:1 - INFO - Current data rate is 64.02 Hz. 19216 events processed
  2015-06-28 23:00:25,171 - master-rnk:1 - INFO - Current data rate is 65.35 Hz. 20417 events processed
  2015-06-28 23:00:39,842 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=21392 took 0.0000 sec
  2015-06-28 23:00:40,759 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.917 sec
  2015-06-28 23:00:40,759 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.930 sec
  2015-06-28 23:00:43,478 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.719 sec
  2015-06-28 23:00:44,161 - master-rnk:1 - INFO - Current data rate is 63.25 Hz. 21618 events processed
  2015-06-28 23:01:02,334 - master-rnk:1 - INFO - Current data rate is 66.09 Hz. 22819 events processed
  2015-06-28 23:01:16,940 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=23769 took 0.0000 sec
  2015-06-28 23:01:17,862 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.922 sec
  2015-06-28 23:01:17,862 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.936 sec
  2015-06-28 23:01:20,633 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.770 sec
  2015-06-28 23:01:21,475 - master-rnk:1 - INFO - Current data rate is 62.74 Hz. 24020 events processed
  2015-06-28 23:01:39,789 - master-rnk:1 - INFO - Current data rate is 65.58 Hz. 25221 events processed
  2015-06-28 23:01:54,139 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=26146 took 0.0000 sec
  2015-06-28 23:01:55,050 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.911 sec
  2015-06-28 23:01:55,050 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.925 sec
  2015-06-28 23:01:57,798 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.748 sec
  2015-06-28 23:01:59,205 - master-rnk:1 - INFO - Current data rate is 61.86 Hz. 26422 events processed
  2015-06-28 23:02:39,819 - master-rnk:1 - INFO - Current data rate is 29.57 Hz. 27623 events processed
  2015-06-28 23:02:53,580 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=28523 took 0.0000 sec
  2015-06-28 23:02:54,524 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.944 sec
  2015-06-28 23:02:54,524 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.960 sec
  2015-06-28 23:02:57,265 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.740 sec
  2015-06-28 23:02:59,191 - master-rnk:1 - INFO - Current data rate is 62.00 Hz. 28824 events processed
  2015-06-28 23:03:17,998 - master-rnk:1 - INFO - Current data rate is 63.86 Hz. 30025 events processed
  2015-06-28 23:03:31,394 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=30900 took 0.0000 sec
  2015-06-28 23:03:32,313 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.919 sec
  2015-06-28 23:03:32,313 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.933 sec
  2015-06-28 23:03:35,140 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.826 sec
  2015-06-28 23:03:37,386 - master-rnk:1 - INFO - Current data rate is 61.95 Hz. 31226 events processed
  2015-06-28 23:03:56,127 - master-rnk:1 - INFO - Current data rate is 64.08 Hz. 32427 events processed
  2015-06-28 23:04:09,192 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=33277 took 0.0000 sec
  2015-06-28 23:04:10,109 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.916 sec
  2015-06-28 23:04:10,109 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.930 sec
  2015-06-28 23:04:12,881 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.772 sec
  2015-06-28 23:04:15,543 - master-rnk:1 - INFO - Current data rate is 61.85 Hz. 33628 events processed
  2015-06-28 23:04:29,895 - server-rnk:96 - INFO - 
  --BEGIN SERVER TIMING--
  ServerTimeToGetData:   31.131ms per event (5759 total calls)
  --END SERVER TIMING--
  2015-06-28 23:04:29,911 - server-rnk:48 - INFO - 
  --BEGIN SERVER TIMING--
  ServerTimeToGetData:   30.613ms per event (5758 total calls)
  --END SERVER TIMING--
  2015-06-28 23:04:29,925 - server-rnk:80 - INFO - 
  --BEGIN SERVER TIMING--
  ServerTimeToGetData:   31.021ms per event (5759 total calls)
  --END SERVER TIMING--
  2015-06-28 23:04:29,981 - server-rnk:32 - INFO - 
  --BEGIN SERVER TIMING--
  ServerTimeToGetData:   34.066ms per event (5761 total calls)
  --END SERVER TIMING--
  2015-06-28 23:04:29,995 - server-rnk:64 - INFO - 
  --BEGIN SERVER TIMING--
  ServerTimeToGetData:   34.749ms per event (5760 total calls)
  --END SERVER TIMING--
  2015-06-28 23:04:30,008 - master-rnk:1 - INFO - master waited for ready servers 7.59 ms per each time. Did 1.00 waits per event
  2015-06-28 23:04:30,008 - master-rnk:1 - INFO - Overall data rate is 59.30 Hz. Number of events is 34550
  2015-06-28 23:04:30,010 - server-rnk:16 - INFO - 
  --BEGIN SERVER TIMING--
  ServerTimeToGetData:   30.736ms per event (5759 total calls)
  --END SERVER TIMING--
  2015-06-28 23:04:30,017 - worker-rnk:2 - INFO - g2worker.calc at 120hz counter=34549 took 0.0000 sec
  2015-06-28 23:04:30,934 - worker-rnk:2 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.917 sec
  2015-06-28 23:04:30,934 - viewer-rnk:0 - INFO - XCorrBase.viewerWorkersUpdate: viewer worker gather communication took: 0.926 sec
  2015-06-28 23:04:30,935 - worker-rnk:2 - INFO - 
  --BEGIN FIRST WORKER TIMING--
  workerWaitForMasterBcastNotWrapped:    2.002ms per cspadevt (34566 total calls)
  workerWaitForMasterBcast:    1.998ms per cspadevt (34566 total calls)
  serverWorkersScatterNotWrapped:    7.451ms per cspadevt (34550 total calls)
  storeNewWorkerDataNotWrapped:    6.983ms per cspadevt (34550 total calls)
  viewerWorkersUpdateNotWrapped:  952.331ms per cspadevt (15 total calls)
  --END FIRST WORKER TIMING--
  2015-06-28 23:04:30,935 - master-rnk:1 - INFO - 
  --BEGIN MASTER TIMING--
  informWorkersOfNewData:    9.204ms per cspadevt (34550 total calls)
  informViewerOfUpdate:    0.027ms per cspadevt (15 total calls)
  --END MASTER TIMING--
  2015-06-28 23:04:33,172 - viewer-rnk:0 - INFO - viewerFormNDarrays took 2.237 sec
  2015-06-28 23:04:44,154 - viewer-rnk:0 - INFO - 
  --BEGIN VIEWER TIMING--
  waitForMasterMessage: 22526.441ms per update (16 total calls)
  viewerWorkersUpdate: 15753.649ms per update (15 total calls)
  --END VIEWER TIMING--


The whole run was about 10 minutes for the 35,000 frames, which is about a 5 minute run. Output reports timing, about 60hz. Output at the end reports timing for 
different pieces of the system. Things to note

* viewerWorkersUpdate: 15753.649ms per update (15 total calls), the viewer is taking 15.8 seconds to finish with all the plots.
* viewerWorkersUpdateNotWrapped:  952.331ms per cspadevt (15 total calls), workers are spending about a second in the gather with the viewer
* ServerTimeToGetData:   34.749ms per event. With six servers, thats on average 5.8ms per event, meaning we can read/calibrate data at 172hz.

Other timing is to try to figure out how well we are keeping up with 120hz data. At 120hz, you have 8.3ms for each event. Ideally, you want the scatter of data from server to workers to take 
up a small part of that so the workers have time to calculate. We aren't seeing that in this run, meaning there is work to do on tuning:

* serverWorkersScatterNotWrapped:    7.451ms per cspadevt (34550 total calls), almost all of the 8.3ms is in the scatter
* storeNewWorkerDataNotWrapped:      6.983ms per cspadevt (34550 total calls), with the G2incremental, this is how much time it takes to update the calculation using 150 cores. This number should go down with more cores.                         
* workerWaitForMasterBcast:          2.002ms per cspadevt (34566 total calls), maybe this can go down?

Those worker numbers total to 16.43, which is close to the 60hz that we are seeing. This is scattering full cspad as 4 byte float32. Masking out some of the cspad pixels can help.

This master timing

*   informWorkersOfNewData:    9.204ms per cspadevt (34550 total calls)

is probably the master waiting for workers to finish with the last event data. However this number

* master-rnk:1 - INFO - master waited for ready servers 7.59 ms per each time. Did 1.00 waits per event

I don't quite understand. masters shouldn't wait this long for a new server - I have to revisit how that number is being reported.



