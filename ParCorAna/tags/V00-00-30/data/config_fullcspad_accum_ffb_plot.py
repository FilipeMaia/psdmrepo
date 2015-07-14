import numpy as np
import psana
import ParCorAna

system_params={}
experiment = 'xcs84213'; run=117  # this has CsPad.DataV2  # 40k - 60k
system_params['dataset'] = 'exp=%s:run=%d:live:stream=0-5:dir=/reg/d/ffb/xcs/%s/xtc' % (experiment, run, experiment) 

#system_params['dataset'] = 'exp=%s:run=%d' % (experiment, run) 

system_params['src']       = 'DetInfo(XcsEndstation.0:Cspad.0)'       # for cspad
system_params['psanaType'] = psana.CsPad.DataV2  # for tutorial data

system_params['ndarrayProducerOutKey'] = 'ndarray'
system_params['ndarrayCalibOutKey'] = 'calibrated'    # set to None to skip NDarrayCalib

system_params['psanaOptions'], \
    system_params['outputArrayType'] = ParCorAna.makePsanaOptions(
                                         srcString=system_params['src'],
                                         psanaType=system_params['psanaType'],
                                         ndarrayOutKey=system_params['ndarrayProducerOutKey'],
                                         ndarrayCalibOutKey=system_params['ndarrayCalibOutKey']
                                       )

system_params['workerStoreDtype'] = np.float32  # The calibration system is creating ndarrays of

system_params['maskNdarrayCoords'] = 'xcs84213-r117_XcsEndstation_0_Cspad_0_mask_based_on_color_ndarrCoords.npy'
system_params['testMaskNdarrayCoords'] = 'xcs84213-r117_XcsEndstation_0_Cspad_0_testmask_ndarrCoords.npy'

system_params['numServers'] = 6
system_params['serverHosts'] = None # system selects which hosts to use

system_params['times'] = 35000     # number of distinct times that each worker holds onto

eventsPerSecond = 120
numSeconds = 30
system_params['update'] = numSeconds*eventsPerSecond # update/viewer publish every n events. 
# Set to 0 to only update at the end.
# This is the number of events the system goes through before doing another viewer publish.


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
user_params['iX'] = 'xcs84213-r117_XcsEndstation_0_Cspad_0_iX.npy'
user_params['iY'] = 'xcs84213-r117_XcsEndstation_0_Cspad_0_iY.npy'

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

