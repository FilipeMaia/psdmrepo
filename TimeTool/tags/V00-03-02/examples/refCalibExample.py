import os
import argparse
import psana

def refCalibExample(datasource, numEvents=0):
    timeToolDebugMessages='TimeTool.Analyze=debug'
    if os.environ.has_key('MSGLOGCONFIG'):
        os.environ['MSGLOGCONFIG'] += ';%s' % timeToolDebugMessages
    else:
        os.environ['MSGLOGCONFIG'] = timeToolDebugMessages

    psanaOptions = {
        ########## psana configuration #################
        'psana.modules':'TimeTool.Analyze TimeTool.PlotAnalyze',

        ########## TimeTool.Analyze configuration #######
        # for getting plot data
        'TimeTool.Analyze.eventdump':1,
        #  Key for fetching timetool camera image
        'TimeTool.Analyze.get_key':'TSS_OPAL',
        #  Results are written to <put_key>
        'TimeTool.Analyze.put_key':'TTANA',
        #  Indicate absence of beam for updating reference
        'TimeTool.Analyze.eventcode_nobeam':0, # beam is never absent
        #  Indicate events to skip (no laser, for example)
        'TimeTool.Analyze.eventcode_skip':0,  # laser is always present
        #  Polynomial coefficients for position_time calculation
        'TimeTool.Analyze.calib_poly':'0 1 0',
        #  Project onto X axis?
        'TimeTool.Analyze.projectX':True,
        #  Minimum required bin value of projected data
        'TimeTool.Analyze.proj_cut':0,
        #  ROI (x) for signal
        'TimeTool.Analyze.sig_roi_x':'0 1023',
        #  ROI (y) for signal
        'TimeTool.Analyze.sig_roi_y':'425 724',
        #  ROI (x) for sideband
        'TimeTool.Analyze.sb_roi_x':'' ,
        #  ROI (y) for sideband
        'TimeTool.Analyze.sb_roi_y':'', 
        #  Rolling average convergence factor (1/Nevents)
        'TimeTool.Analyze.sb_avg_fraction':0.05,
        #  Rolling average convergence factor (1/Nevents)
        'TimeTool.Analyze.ref_avg_fraction':0.05,  # we will be loading ref from calib, so this will not be used
        #  Read weights from a text file
        'TimeTool.Analyze.weights_file':'',
        #  Indicate presence of beam from IpmFexV1::sum() [monochromator]
        'TimeTool.Analyze.ipm_get_key':'',
        # 'TimeTool.Analyze.ipm_beam_threshold':'',

        # look up initial reference from calibration, based on run number
        'TimeTool.Analyze.use_calib_db_ref':1,
        #  Load initial reference from file
        'TimeTool.Analyze.ref_load':'',
        #  Save final reference to file
        'TimeTool.Analyze.ref_store':'',
        #  Generate histograms for initial events, dumped to root file (if non zero)
        'TimeTool.Analyze.dump':0, # make sure this is 0 if using MPI, otherwise race conditions with root files
        #  Filter weights
        'TimeTool.Analyze.weights':'0.00940119 -0.00359135 -0.01681714 -0.03046231 -0.04553042 -0.06090473 -0.07645332 -0.09188818 -0.10765874 -0.1158105  -0.10755824 -0.09916765 -0.09032289 -0.08058788 -0.0705904  -0.06022352 -0.05040479 -0.04144206 -0.03426838 -0.02688114 -0.0215419  -0.01685951 -0.01215143 -0.00853327 -0.00563934 -0.00109415  0.00262359  0.00584445  0.00910484  0.01416929  0.0184887   0.02284319  0.02976289  0.03677404  0.04431778  0.05415214  0.06436626  0.07429347  0.08364909  0.09269116  0.10163601  0.10940983  0.10899065  0.10079016  0.08416471  0.06855799  0.05286105  0.03735241  0.02294275  0.00853613',
    }
    psana.setConfigFile("")
    psana.setOptions(psanaOptions)
    ds = psana.DataSource(datasource)

    for idx,evt in enumerate(ds.events()):
        if numEvents>0 and idx >= numEvents:
            break

DEFAULT_DATASOURCE = 'exp=sxrd5814:run=150'
    

programDescription = '''
example that uses the use_calib_db_ref option of TimeTool.Analyze to look up
a reference image from the calib store.

It assumes an experiment and run where the laser is always on. 

Default datasource is %s, but may not be available
'''  % DEFAULT_DATASOURCE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=programDescription, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', '--numevents', type=int, help="number of events to process, default is all", default=0)
    parser.add_argument('-d', '--datasource', type=str, help="set datasource string (replace default test data)", default=DEFAULT_DATASOURCE)

    args = parser.parse_args()

    refCalibExample(args.datasource, args.numevents)
