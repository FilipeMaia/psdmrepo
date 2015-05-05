import numpy as np
import psana
import ParCorAna

###############################################################
##  This is a params file for parCorAnaDriver
## 
##  Use it to define two dictionaries
##  system_params - parameters that will be used by the framework
##  user_params   - parameters that will be passed to the user G2 module

############## system_params #####################

system_params={}

######## dataset ########### 
# specify the experiment/run. This is used to define the dataset, and h5output file.

run = 1437
experiment = 'xpptut13'

# some other examples for testing
# experiment = 'xcsc9114'; run=20  # this is for testing. No real signal, but a very long run to test performance
# experiment = 'xcsi0314'; run=211  # 63k frames of epix, mask all but bottom quarter, run 214 is dark
# experiment = 'xcsi0314'; run=177  # cspad2x2, dark is 179
# experiment = 'xcsi0314'; run=178  # cspad2x2, dark is 179
# experiment = 'xcs84213'; run=117  # this has CsPad.DataV2  # 40k - 60k

system_params['dataset'] = 'exp=%s:run=%d' % (experiment, run) 

# below, example where one specifies 
# system_params['dataset'] = 'exp=%s:run=%d:live:dir=/reg/d/ffb/xcs/xcs84213' % (experiment, run)  # tutorial data for getting started

# system_params['dataset'] = 'shmem=XCS:stop=no'     # when running on shared memory, may want to set h5 ouput to None for shmem

############### SRC TYPE PSANA CONFIG #################
system_params['src']       = 'DetInfo(XppGon.0:Cspad.0)' # for tutorial data

# some common xcs sources 
#system_params['src']       = 'DetInfo(XcsEndstation.0:Cspad2x2.0)'    # for cspad2x2
#system_params['src']       = 'DetInfo(XcsEndstation.0:Cspad.0)'       # for cspad
#system_params['src']       = 'DetInfo(XcsEndstation.0:Epix100a.0)'    # for Epix100a.0

system_params['psanaType'] = psana.CsPad.DataV2  # for tutorial data

#system_params['psanaType'] = psana.CsPad2x2.ElementV1 
#system_params['psanaType'] = psana.Epix.ElementV2

############## PSANA CONFIG OPTIONS / OUTPUT ARRAY TYPE / OUTPUT KEY ##############
# Create psana options to calibrate detector data. Uses the ParCorAna.makePsanaOptions 
# utility function. One does:
#
#   * specify the type and src for the detector (above), 
#   * specify output keys for the two modules (below)
#   * save the resulting options and final ndarray type produced in the system_params dict. (see below). 
#
# The options can be extended after calling this function.
# To see the final options, just run this file as a standalone python script.
#
# Users can also skip calibration and just work with the NdArrayProducer output.
# 

system_params['ndarrayProducerOutKey'] = 'ndarray'
system_params['ndarrayCalibOutKey'] = 'calibrated'    # set to None to skip NDarrayCalib

system_params['psanaOptions'], \
    system_params['outputArrayType'] = ParCorAna.makePsanaOptions(
                                         srcString=system_params['src'],
                                         psanaType=system_params['psanaType'],
                                         ndarrayOutKey=system_params['ndarrayProducerOutKey'],
                                         ndarrayCalibOutKey=system_params['ndarrayCalibOutKey']
                                       )

# an example of adding options to the psana configuration:
# system_params['psanaOptions']['ImgAlgos.NDArrCalib.do_gain'] = True

system_params['workerStoreDtype'] = np.float64  # The calibration system is creating ndarrays of
              # double - or np.float64. Each worker stores a portion of this ndarray. To guarantee no 
              # loss of precision, workers should store results in the same data format - i.e, float64.
              # However for large detectors and long correlation types, this may require too much 
              # memory. For full cspad where all pixels are included in the mask, and 50,000 times are stored
              # on the workers, this amounts to 50,000*(32*388*185)*8=855GB of memory that must be 
              # distributed amoung all the workers. If each host has 24GB (as it is presently), one would 
              # have to use 36 hosts. Given that each host runs 12 MPI ranks, we need 432 ranks for the workers.
              #
              # A simple way to use less memory, is to have the workers store the detector data as 4
              # byte floats, or change from np.float64 to np.float32. One could also try np.int16 to
              # get down to 2 bytes. It is not reccommended that one use unsigned integers as calibration
              # can produce negative values. Note that the delay curves will always be calculated with float64
              # in order to avoid loss of precision with that calculation. Changing the worker_store_type 
              # does not affect the precision of the delay curves.
              #
              # The system will emit warnings if the calibrated ndarrays have to be truncated to fit
              # into a workerStoreDtype that is not np.float64.


############## mask ##############
# The mask a numpy array of int's that must have the same shape as the detector array returned by
# the NDArr converter psana module. It must be 1 for elements to process, and 0 for elements not to process.
# Use the tool: mask_color_tool to create mask and color .npy files and convert between image and 
# ndarray coords
#
# note: a mask file must be provided.

system_params['maskNdarrayCoords'] = "see tutorial for documentation on this parameter"
system_params['testMaskNdarrayCoords'] = None

####### numservers serverHosts #################
# The servers are responsible for working through the data. If they do a lot of 
# processing of the detector data, such as complicated calibration, they can become a bottleneck.
# Running several servers in parallel can increase performance. The 'numservers' parameter allows
# users to increase the number of servers. When running from shared memory, only certain nodes have 
# access to the data. In this case, users need to specify the names of these hosts. This is what the
# 'serverHosts' parameter is for. 
#
# There are limits on the number of servers on can use. These depend on the input mode specified in the
# psana datasource string. When running in live mode, the most servers one can run is the number of 
# distinct DAQ streams (usually six). When running in indexing mode, there is no limit, but more 
# servers creates more I/O. Performance in index mode has not been evaluated. When running from shared 
# memory, there is no I/O limit on the number of servers. 
#
# The big limit on the number of servers is the balance of how many MPI ranks are servers vs. workers.
# Ideally, one uses as many workers as possible, while using enough servers to read through the data.
#
# For testing locally, it is reccommended that you use 1 server. 
#
# When running in live mode on the batch farm, it is recommended that you use the same number of 
# servers as you have DAQ streams. Most always this is 6. In addition, it is recommended that
# you run an MPI job with at least 12 * (numservers-1) + 1 procs - or at least 61. The framework will choose
# MPI ranks on distinct nodes when it can. It seems that having servers be on distinct nodes increases
# performance. Presently, each psana node has 12 cores, once you use 61 cores you're job will cover at 
# least 6 nodes.
#
# To see exactly how many DAQ streams you have, look at the files for your run. For example
#
#  ls -lrth /reg/d/psdm/xpp/xpptut13/xtc/*-r1437*.xtc

#  -r--r-x---+ 1 psdatmgr ps-data 9.4G Oct 20  2013 /reg/d/psdm/xpp/xpptut13/xtc/e308-r1437-s00-c00.xtc
#  -r--r-x---+ 1 psdatmgr ps-data 9.4G Oct 21  2013 /reg/d/psdm/xpp/xpptut13/xtc/e308-r1437-s02-c00.xtc
#  -r--r-x---+ 1 psdatmgr ps-data 9.4G Oct 21  2013 /reg/d/psdm/xpp/xpptut13/xtc/e308-r1437-s01-c00.xtc
#
# Shows that there are only 3 streams for run 1437 of the xpp tutorial data. Note, there is a difference
# between streams that are numbered at 80+ vs streams numbered less than 80. 80+ are control streams, and
# streams less than 80 are DAQ streams. Do not count control streams, only count DAQ streams. For live
# mode, do not use more streams than there are DAQ streams.

system_params['numServers'] = 1
system_params['serverHosts'] = 1

# when explicitly listing hosts, the number of hosts much agree with the number of servers
# system_params['serverHosts'] = ['host1', 'host2']  # to explicitly set hosts, use a list of hostnames

# here are the hosts used for each instrument in shared memory mode
# system_params['serverHosts'] = ['daq-amo-mon01', 'daq-amo-mon02', 'daq-amo-mon03'] 
# system_params['serverHosts'] = ['daq-xcs-mon01', 'daq-xcs-mon02', 'daq-xcs-mon03']
# system_params['serverHosts'] = ['daq-xpp-mon01', 'daq-xpp-mon02', 'daq-xpp-mon03']
# system_params['serverHosts'] = ['daq-cxi-mon01', 'daq-cxi-mon02', 'daq-cxi-mon03']
# system_params['serverHosts'] = ['daq-sxr-mon01', 'daq-sxr-mon02', 'daq-sxr-mon03']
# system_params['serverHosts'] = ['daq-mec-mon01', 'daq-mec-mon02']


########### times ######## update ##############
# to change or add to the default configuration, one can do things like
# system_params['psanaOptions']['ImgAlgos.NDArrCalib.do_cmod']=False  # turn off common mode correction

system_params['times'] = 50000     # number of distinct times that each worker holds onto

eventsPerMinute = 120*60
numMinutes = 4
system_params['update'] = eventsPerMinute*numMinutes  # update/viewer publish every n events. 
              # Set to 0 to only update at the end.
              # This is the number of events the system goes through before doing another viewer publish. Note - 
              # the workers, on 250 cores, will takes 130 seconds when processing 50,000 events.


######### delays ############
system_params['delays'] = ParCorAna.makeDelayList(start=1,
                                                  stop=25000, 
                                                  num=100,
                                                  spacing='log',  # can also be 'lin'
                                                  logbase=np.e)

######## User Module ########
import ParCorAna.UserG2 as UserG2
system_params['userClass'] = UserG2.G2atEnd

######## h5output, overwrite ########### 
# The system will manage an h5output filename. This is not a file for collective
# writes. Within the user code, only the viewer rank should write to the file. The viewer
# will receive an open group to the file at run time.
# In particular, for G2, the software will save the G2, IP IF 3D matrices, delays and counts to an h5 file.

# set h5output to None if you do not want h5 output.
# The system will recognize %T in the filename and replaces it with the current time in the format
# yyyymmddhhmmss. (year, month, day, hour, minute, second). It will also recognize %C for a one up counter.
# When forming a string by "a-%d" % 3, one needs to use two % to  Add %%T into the string
system_params['h5output'] = 'g2calc_%s-r%4.4d.h5' % (experiment, run)
system_params['testh5output'] = 'g2calc_test_%s-r%4.4d.h5' % (experiment, run)

## overwrite can also be specified on the command line, --overwrite=True which overrides what is below
system_params['overwrite'] = False   # if you want to overwrite an h5output file that already exists

######## verbosity #########
# verbosity can be one of INFO, DEBUG, WARNING (levels from the Python logging module)
system_params['verbosity'] = 'INFO'

######## numevents #########
# numevents - 0 or None means all events. This is primarily a debugging/development switch that
# can be overriden on the command line
system_params['numEvents'] = 0
system_params['testNumEvents'] = 100

######## elementsperworker #########
# elementsperworker - 0 or None means all elements. This is a debugging/development switch only.
# use the maskNdarrayCoords to control which elements are processed
# this can be overriden on the command line
system_params['elementsPerWorker'] = 0

##### parallel job allocation #####
# below are parameters that the system will use to figure out the 
# command to use to launch the job - MOVE INTO DRIVER
#system_params['GB_per_host'] = 23.53   
# there are a few hosts out there with 19.59GB - psana1116 in psanaq, psana1305 in psfehq
# number_hosts_in_queue = {'psanaq':40,        # 941GB
#                         'psfehpriorq':16,   # 372GB 
#                         'psnehpriorq':16}   # 376GB 
#system_params['slots_per_host'] = 12
#system_params['number_hosts_in_queue'] = 40

system_params['queue'] = 'psanaq' # put None for local
system_params['bsubCmd'] = None  # let the system construct bsub_cmd

##################################################
############ USER MODULE - G2 CONFIG #############
user_params = {}

# the partition is a numpy array of int's. 0 and negative int's are ignored. int's that are positive
# partition the elements. That is all elements with '1' form one delay curve, likewise all elements that are '2'
# form another delay curve.
user_params['colorNdarrayCoords'] =  "see __init__.py in ParCorAna/src for documentation on this file"
user_params['saturatedValue'] = (1<<15)
user_params['LLD'] = 1E-9
user_params['notzero'] = 1E-5


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

