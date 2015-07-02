#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module RegDBUtils.py...
#
#------------------------------------------------------------------------

"""RegDB Global methods

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision$"
# $Source$


#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
from time import time # for test purpose only
#from time import localtime, gmtime, strftime, clock, time, sleep
#import numpy as np

from RegDB import experiment_info

#------------------------------

def unique_detector_names() :
    """Returns complete list of available detectors at LCLS"""
    return experiment_info.unique_detector_names()

#------------------------------

def detectors (ins, exp, run, style='psana') :
    """Returns the list of detectors, for example: ['BldEb-0|NoDevice-0', 'CxiDg1-0|Tm6740-0', 'CxiDg2-0|Tm6740-0', 'CxiDs1-0|Cspad-0', ...].
    """

    if style == 'psana' : return list_of_sources_in_run(ins, exp, run)
    else :                return experiment_info.detectors(ins, exp, run)

#------------------------------

def list_of_sources_in_run (ins, exp, run) :
    """Returns the list of detectors in style of psana, for example 'CxiDs1.0:Cspad.0'
    """
    return [det.replace("|",":").replace("-",".") for det in experiment_info.detectors (ins, exp, run)]

#------------------------------

def txt_of_sources_in_run (ins, exp, run) :
    """Returns the list of detectors as formatted text with heading line
    """
    list_of_detectors = detectors(ins, exp, run)
    txt = '\nList of detectors for inst: %s  exp: %s  run %d' % (ins, exp, run)
    for det in list_of_detectors :
        txt += '\n'+det
    return txt

#------------------------------

def list_of_sources_in_run_for_selected_detector (ins, exp, run, det_selected) :
    """Returns the list of detectors in run for selected detector. For example, for CSPAD returns ['CxiDs1.0:Cspad.0', 'CxiDsd.0:Cspad.0']
    """
    pattern = det_selected.lower() + '.'
    return [src for src in list_of_sources_in_run(ins, exp, run) if src.lower().find(pattern) != -1]

#------------------------------

def list_of_sources_for_det (det_name='CSPAD') :
    """Returns the list of sources for specified detector in style of psana, for example 'CxiDs1.0:Cspad.0'
    """
    pattern = det_name.lower() + '-'
    lst = [src.replace("|",":").replace("-",".") for src in experiment_info.unique_detector_names() if src.lower().find(pattern) != -1]
    #lst = [src.replace("|",":").replace("-",".") for src in experiment_info.unique_detector_names() if src.lower().find(pattern) != -1 and src.find('NoDet') == -1]
    #for i, det in enumerate(lst) : print '%4d : %s' %(i, det)
    return lst

#------------------------------





#------------------------------

def experiment_runs (ins, exp) :
    """Returns the list of dictionaries, one dictionary per run, containing run parameters,
    for example: {'begin_time_unix': 1375417636, 'end_time_unix': 1375417646, 'num': 1L, 'exper_id': 329L, 'end_time': 1375417646535192694L, 'begin_time': 1375417636042155759L, 'type': 'DATA', 'id': 69762L} 
    """
    return experiment_info.experiment_runs(ins, exp)

#------------------------------

def dict_run_type (ins, exp) :
    """Returns the dictionary of pairs run:type,
    for example: {1: 'DATA', 2: 'DATA', 3: 'DATA', 4: 'DATA', 5: 'DATA',...}
    """
    runs = experiment_info.experiment_runs(ins, exp)
    dict_rt = {}
    for i,rec in enumerate(runs) :
        num, type = int(rec['num']), rec['type']
        #print i, rec, num, type
        dict_rt[num] = type
    return dict_rt

#------------------------------

def list_of_runnums (ins, exp) :
    """Returns the list of run numbers for specified experiment.
    """
    runs = experiment_info.experiment_runs(ins, exp)
    lst = []
    for rec in runs :
        lst.append( int(rec['num']) )
    return lst

#------------------------------

def list_of_runstrings (ins, exp) :
    """Returns the list of run numbers for specified experiment.
    """
    runs = experiment_info.experiment_runs(ins, exp)
    lst = []
    for rec in runs :
        lst.append( '%04d'%rec['num'] )
    return lst

#------------------------------

def run_attributes (ins, exp, run) :
    """Returns the list of dictionaries, one dictionary per attribute.   
    Example of dict.: {'val': None, 'type': 'TEXT', 'class': 'Calibrations', 'descr': '', 'name': 'comment'}.
    """
    t0_sec = time()
    list_of_dicts = experiment_info.run_attributes(ins, exp, run)
    #print 'run_attributes for %s %s run:%d, t(sec) = %f' % (ins, exp, run, time()-t0_sec)
    return list_of_dicts

#------------------------------

def print_run_attributes (ins, exp, run) :
    """Prints run attributes in formatted table 
    """
    for attr in run_attributes(ins, exp, run) :
        print 'class: %s  name: %s  type (of the value): %s  value (optional): %s  description (optional):%s' \
              % (attr['class'].ljust(16), attr['name'].ljust(32), attr['type'].ljust(8), str(attr['val']).ljust(8), attr['descr'].ljust(8))

#------------------------------

def is_calibration_tag_for_name(ins, exp, run, name='dark') :
    """Returns True if run has a tag for specified name in class 'Calibrations' and
    False othervise
    """
    for attr in run_attributes(ins, exp, run) :
        if attr['class'] == 'Calibrations' and attr['name'] == name : return True
    return False

#------------------------------

def dict_runnum_dark (ins, exp, list_of_nums=None) :
    """Returns dictionary of pairs runnum:is_dark for runs in specified experiment.
    By default - for all runs in the experiment,
    if the list_of_nums is provided - for specified runs. 
    """
    t0_sec = time()
    list_of_rn = list_of_nums
    if list_of_nums is None : list_of_rn = list_of_runnums (ins, exp)
    dict_rnd ={}
    for runnum in list_of_rn :
        dict_rnd[runnum] = is_calibration_tag_for_name(ins, exp, runnum, name='dark')
    print '\n\ndict_runnum_dark consumed time (sec) =', time()-t0_sec
    return dict_rnd

#------------------------------
#------------------------------
#------------------------------
#------------------------------

def calibration_runs (ins, exp) :
    """Returns dictionary for pairs {runnum:dict_rec},
    where dict_rec is a dictionary with info fields, for example: {'comment': '', 'calibrations': []}
    """
    return experiment_info.calibration_runs(ins, exp)

#------------------------------

def print_calibration_runs (ins, exp) :
    calibrans = calibration_runs (ins, exp)
    for r,v in calibrans.iteritems() :
        print r,v
    #print calibrans

#------------------------------

def dict_of_recs_for_run (ins, exp, runnum) :
    """Returns dictionary of records for run number
    For example, {'comment': 'Dark as reported by Philip', 'calibrations': ['dark']}
    """
    return calibration_runs(ins, exp)[runnum]

#------------------------------

def comment_for_run (ins, exp, runnum) :
    """Returns comment for run number
    For example, 'Dark as reported by Philip'
    """
    return dict_of_recs_for_run(ins, exp, runnum)['comment']

#------------------------------

def list_of_calibs_for_run (ins, exp, runnum) :
    """Returns list of calibration names for run
    For example, ['dark','flat']
    """
    return dict_of_recs_for_run(ins, exp, runnum)['calibrations']

#------------------------------

def print_unique_detector_names () :
    for i, detname in enumerate(unique_detector_names()):
        print '%4d : %s' %(i, detname)

def print_list_of_sources_for_det (det_name='CSPAD') :
    for i, detname in enumerate( list_of_sources_for_det (det_name)) :
        print '%4d : %s' %(i, detname)
 
#------------------------------
#------------------------------
#------------------------------
#------------------------------

if __name__ == "__main__" :

    print "\n\nTest detectors('XPP', 'xppa4513', 1):" 
    print detectors('XPP', 'xppa4513', 1)

    print "\n\nTest experiment_runs('XPP', 'xppa4513'):"
    print experiment_runs('XPP', 'xppa4513')

    print "\n\nTest dict_run_type('CXI', 'cxic0213'):"
    print dict_run_type('CXI', 'cxic0213')

    print "\n\nTest run_attributes('CXI', 'cxic0213', 215):"
    print run_attributes('CXI', 'cxic0213', 215)

    print "\n\nTest print_run_attributes('CXI','cxic0213', 215):" 
    print_run_attributes('CXI','cxic0213', 215)

    print "\n\nTest detectors('CXI','cxic0213', 215):" 
    print detectors('CXI','cxic0213', 215)

    l = list_of_runnums('CXI','cxic0213')
    print "\n\nTest list_of_run_nums('CXI','cxic0213'):\n", l 

    d = dict_runnum_dark ('CXI','cxic0213',list_of_nums=[1,163,206,220])
    print "Test dict_runnum_dark ('CXI','cxic0213',list_of_runnums=[1,163,206,220]):\n", d

    d = dict_runnum_dark ('CXI','cxic0213')
    print "Test dict_runnum_dark ('CXI','cxic0213'):\n", d

    print "\n\nTest print_calibration_runs('CXI','cxic0213'):" 
    print_calibration_runs('CXI','cxic0213')

    print "\n\nTest : dict_of_recs_for_run ('CXI','cxic0213, 162)"
    print dict_of_recs_for_run ('CXI','cxic0213', 162)

    print "\n\nTest : comment_for_run ('CXI','cxic0213', 162)"
    print comment_for_run ('CXI','cxic0213',162)

    print "\n\nTest : list_of_calibs_for_run('CXI','cxic0213', 162)"
    print list_of_calibs_for_run('CXI','cxic0213',162)

    print "\n\nTest : unique_detector_names()"
    print_unique_detector_names()

    l = list_of_runnums('XPP','xppc7014')
    print "\n\nTest list_of_run_nums('XPP','xppc7014'):\n", l 

    #det_name = 'cspad'
    #det_name = 'cspad2x2'
    #det_name = 'epix'
    #det_name = 'epix100a'
    det_name = 'andor'
    #det_name = 'epix10k'
    #det_name = 'princeton'
    #det_name = 'pnccd'
    #det_name = 'tm6740'
    #det_name = 'opal1000'
    #det_name = 'opal2000'
    #det_name = 'opal4000'
    #det_name = 'opal8000'
    #det_name = 'orcafl40'
    #det_name = 'fccd960'
    #det_name = 'Acqiris'
    print "\n\nTest : list_of_sources_for_det ('%s')" % det_name
    print_list_of_sources_for_det (det_name)
    sys.exit ( "End of test" )

#------------------------------
