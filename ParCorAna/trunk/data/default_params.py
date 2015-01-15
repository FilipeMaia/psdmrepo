from mpi4py import MPI
import logging
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
# specify the experiment/run or shared memory. Any psana dataset string can be used, options
# like live, shmem, idx, run, stream can be used. The software will do different things depending 
# on the dataset options.

run = 1437
experiment = 'xpptut13'

# some other examples for testing
# experiment = 'xcsc9114'; run=20  # this is for testing. No real signal, but a very long run to test performance
# experiment = 'xcsi0314'; run=211  # 63k frames of epix, mask all but bottom quarter, run 214 is dark
# experiment = 'xcsi0314'; run=177  # cspad2x2, dark is 179
# experiment = 'xcsi0314'; run=178  # cspad2x2, dark is 179
# experiment = 'xcs84213'; run=117  # this has CsPad.DataV2  # 40k - 60k

system_params['dataset'] = 'exp=%s:run=%d' % (experiment, run)  # tutorial data for getting started

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
# below we set the psana options/config that load two psana modules: 
#   an ndarry producer and a calibration module. 
# Using the ParCorAna.makePsanaOptions, the user just specifies the type and src for the dector (above), 
# uses default output keys for the two modules (below), and save the resulting options and and final ndarray 
# type produced in the system_params dict. (see below). The options can be extended after calling this function.
# To see the final options, just run this file as a standalone python script.
#
# TypeNdArrayProducer        ->   NdArrayCalib   
# (make ndarray of double)        (produces ndarray of the same size)
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

############## mask ##############
# The mask a numpy array of int's that must have the same shape as the detector array returned by
# the NDArr converter psana module. It must be 1 for elements to process, and 0 for elements not to process.
# Use the tool: mask_color_tool to create mask and color .npy files and convert between image and 
# ndarray coords
#
# note: a mask file must be provided.

system_params['mask_ndarrayCoords'] = "see __init__.py in ParCorAna/src for documentation on this parameter"

####### numservers serverhosts #################
#
# The server(s) calibrates the detector data and scatters it to the workers. 
# The user module do processing to filter what events are scattered. The servers
# can become a bottleneck. You want just enough servers to keep up with the data
# rate, but not so many that you take away resources from the workers. 
#
# in index mode, it is easy to add more servers, the index files are used to
# distribute the events.
#
# in shared memory mode, servers have to run on hosts with shared memory slices.
# These shared memory slices distribute the events, and each server only serves an
# event once, so it is easy to distribute the events among the servers - multiple 
# framework servers can request events from the same shared memory server and not get
# the same event. However to cover all the events, you want at least one server on each
# daq monitoring node with a shared memory server (Assuming these servers partition the
# events among themselves).  The framework knows the daq monitoring nodes for each 
# instrument. By default it will distribute the servers among these hosts in a round
# robin fashion, but it presently does not check if shared memory servers are running
# on those nodes. It assumes all monitoring nodes get a shared memory slice. Users can 
# specify a host list to override this default behavior. The system will pick ranks on 
# these hosts for the framework servers.
#
# in live & offline linear mode, it is more difficult to distribute events (note, there 
# should be no point to running offline linear mode, use indexing mode instead). To do so,
# each server handles a fraction of the streams. Typically there are 6 DAQ streams and
# 1 or 2 IOC control streams. The IOC control streams with xtcav and other data require 
# special handling. Since these streams (numbered 80 and above) have data that 
# needs to be merged with all the other streams, they must be passed to each server.
#
# For example, if there are 6 streams, 0,1,2,3,4,5 and a stream 80, then if numservers = 3 
# the stream assignment will be
#
# server 0: streams 0,3,80
# server 1: streams 1,4,80
# server 2: streams 2,5,80
#
# If 80 is excluded from the streams via the :stream=0-6 command in the datasource string, then 
# none of the servers will read stream 80 and one should expect better performance.
#
# If one is scattering a camera from s80 to the workers in live mode, one should probably 
# only use one server.
#
system_params['numservers'] = 1
system_params['serverhosts'] = None  # None means system selects which hosts to use (default). 
# system_params['serverhosts'] = ['host1', 'host2']  # to explicitly set hosts, use a list of hostnames


########### times ######## update ##############
# to change or add to the default configuration, one can do things like
# system_params['psanaOptions']['ImgAlgos.NDArrCalib.do_cmod']=False  # turn off common mode correction

system_params['times'] = 50000     # number of distinct times that each worker holds onto


system_params['update'] = 120*60*4  # update/viewer publish every n events. 
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
system_params['user_class'] = UserG2.UserG2

######## h5output, overwrite ########### 
# The system will manage an h5output filename. This is not a file for collective
# writes. Within the user code, only the viewer rank should write to the file. 
# The system master rank may append to the file after running the viewer code.
#
# In particular, for G2, the software will save the G2, IP IF 3D matrices, delays and counts to an h5 file.

# set h5output to None if you do not want h5 output.

# The code below constructs an h5output argument based on the dataset string.
# It uses exp= and run= to construct the default name.
# if using shared memory, these are not available. Remove default behavior (which throws
# assert exceptions if 'exp' and 'run' are not in the dataset string) and either hard code
# h5output filename, or use the command line switch -o to set an h5output filename (overrides 
# this config file)

system_params['h5output'] = 'g2calc_%s-r%4.4d.h5'% (experiment, run)

## overwrite can also be specified on the command line, --overwrite=True which overrides what is below
system_params['overwrite'] = False   # if you want to overwrite an h5output file that already exists

######## verbosity #########
# verbosity can be one of logging.INFO, logging.DEBUG, logging.WARNING
# this can be overriden on the command line with aliases for those Python logging module values.
system_params['verbosity'] = logging.INFO 

######## numevents #########
# numevents - 0 or None means all events. This is primarily a debugging/development switch that
# can be overriden on the command line
system_params['numevents'] = 0

######## elementsperworker #########
# elementsperworker - 0 or None means all elements. This is a debugging/development switch only.
# use the mask_ndarrayCoords to control which elements are processed
# this can be overriden on the command line
system_params['elementsperworker'] = 0


##################################################
############ USER MODULE - G2 CONFIG #############
user_params = {}

# the partition is a numpy array of int's. 0 and negative int's are ignored. int's that are positive
# partition the elements. That is all elements with '1' form one delay curve, likewise all elements that are '2'
# form another delay curve.
user_params['color_ndarrayCoords'] =  "see __init__.py in ParCorAna/src for documentation on this file"
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

