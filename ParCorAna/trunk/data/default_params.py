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
# specify the experiment/run.  This is used to define the dataset, and h5output file.

run = 1437
experiment = 'xpptut13'

# below are some datasets used for testing and development of ParCorAna
# experiment = 'xcsc9114'; run=20  # this is for testing. No real signal, but a very long run to test performance
# experiment = 'xcsi0314'; run=211  # 63k frames of epix, mask all but bottom quarter, run 214 is dark
# experiment = 'xcsi0314'; run=177  # cspad2x2, dark is 179
# experiment = 'xcsi0314'; run=178  # cspad2x2, dark is 179
# experiment = 'xcs84213'; run=117  # this has CsPad.DataV2  # 40k - 60k

# when setting the dataset, explicitly set the stream if the DAQ streams are not 0-n, see below for example
system_params['dataset'] = 'exp=%s:run=%d' % (experiment, run) 

# for online monitoring against live data, specify :live and the ffb directory, for example:
# system_params['dataset'] = 'exp=%s:run=%d:live:dir=/reg/d/ffb/xcs/xcs84213' % (experiment, run) 

# when using numservers > 1, set it to exactly the number of DAQ streams (usually 6). If doing
# a run with a different arrangement of streams, explictly list them, for example:
# system_params['dataset'] = 'exp=%s:run=%d:live:stream=1,2,5,80,81:dir=/reg/d/ffb/xcs/xcs84213' % (experiment, run)# and set numservers (below) to 3 

# note, the use of explicitly setting stream. If you set numservers to the  - the system cannot figure out the streams

# example for shared memory, however working against shared memory is not tested:
# system_params['dataset'] = 'shmem=XCS:stop=no'


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
# utility function. 
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

system_params['workerStoreDtype'] = np.float32  # The calibration system is creating ndarrays of
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
              # need not affect the precision of correlation calculations. One can convert the narrower
              # stored values to float64 before calculating for a specific delay.
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
# The servers are responsible for working through the data, breaking up an ndarray of detector 
# data, and scattering it to the workers. When developing, we usuaully specify 
# one server. When analyzing data in live mode, we usually specify 6 servers, or however many
# DAQ streams there are in the run. The framework sets things up so that each server only processes
# one stream. As long as each server can run at 20hz it will keep up with live 120hz data. 
# If you are analyzing xtcav data, then each server will process 2 or more streams. The framework 
# outputs timing at the end which gives us an idea of how fast or slow the servers are.
# Specifying more than 6 servers will not help, rather it will waste too many ranks on servers.
#
# In index mode, specifying more than six servers can help the servers run faster. However usually
# the bottleneck will be with the workers, and more than six servers is not neccessary.
#
system_params['numServers'] = 1
system_params['serverHosts'] = None # system selects which hosts to use

# explicitly listing hosts for servers is only useful when running against hared memory. The 
# number of server hosts must aggree with the numServers parameter.

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

eventsPerSecond = 120
numSeconds = 20
system_params['update'] = numSeconds * eventsPerSecond  # update/viewer publish every n events. 
              # Set to 0 to only update at the end.


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
# The system will optionally manage an h5output file. This is not a file for collective MPI
# writes. Within the user code, only the viewer rank should write to the file. The viewer
# will receive an open group to the file at run time. 
#
# set h5output to None if you do not want h5 output - important to speed up online monitoring with 
# plotting.
#
# The system will recognize %T in the filename and replaces it with the current time in the format
# yyyymmddhhmmss. (year, month, day, hour, minute, second). It will also recognize %C for a three
# digit one up counter. When %C is used, it looks for all matching files on disk, selects the
# one with the maximum counter value, and adds 1 to that for the h5output filename.
#
# Testing is built into the framework by allowing one to run an alternative calculation
# that receives the same filtered and processed events at the main calculation. When the
# alternative calcuation is run, the framework uses the testh5output argument for the
# filename.

system_params['h5output'] = 'g2calc_%s-r%4.4d.h5' % (experiment, run)
system_params['testH5output'] = 'g2calc_test_%s-r%4.4d.h5' % (experiment, run)


# example of using %T and %C, note the %% in the value to get one % in the string after 
# expanding experiment and run:

# system_params['h5output'] = 'g2calc_%s-r%4.4d_%%T.h5' % (experiment, run)
# system_params['h5output'] = 'g2calc_%s-r%4.4d_%%C.h5' % (experiment, run)

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

##################################################
############ USER MODULE - G2 CONFIG #############
user_params = {}

# the below 'color' file must be a .npy file that partitions the ndarray elements
# into sets of pixels that are averaged together to make the delay curves.
# value 0 and negative int's are ignored. int's that are positive
# partition the elements. That is all elements with '1' form one delay curve, likewise all elements that are '2'
# form another delay curve.
user_params['colorNdarrayCoords'] =  "see tutorial for documentation on this parameter"

# The finecolor file below is like the above, a .npy array with the same dimension as the 
# detector data (and mask file). This partition is used to replace each pixel in the IP and IF
# matricies with its average on the color it is in - this is done before doing the final G2 
# calculation to make the delay curve points
user_params['colorFineNdarrayCoords'] =  "see tutorial for documentation on this parameter"

user_params['saturatedValue'] = (1<<15)
user_params['LLD'] = 1E-9
user_params['notzero'] = 1E-5
user_params['psmon_plot'] = False
# to set a different port for psmon plotting, change this
# user_params['psmon_port'] = 12301
user_params['plot_colors'] = None
user_params['print_delay_curves'] = False
user_params['debug_plot'] = False
user_params['iX'] = None
user_params['iY'] = None

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

