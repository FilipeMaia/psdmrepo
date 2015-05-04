####!/usr/bin/env python
#--------------------
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CorAnaPars...
#
#------------------------------------------------------------------------

""" Project: Evaluation of the Image Auto-Correlation Function
CorAnaPars work as a part of the python file managing script in
the process of evaluation of the Image Auto-Correlation Function.

CorAnaPars is a singleton class object to hold common parameters.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule: CorAnaSubmit.py CorAnaPars.py CorAna*.cpp

@version $Id: 2012-09-26 15:00:00Z dubrovin$

@author Mikhail Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 1 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

import os
import sys
import numpy as np
#import matplotlib.pyplot as plt

#--------------------

class CorAnaPars :
    """This is a singleton class for common parameters

    @see BaseClass
    @see OtherClass
    """
    cmd_split    = None # 'psana'
    cmd_proc     = None # 'corana'
    cmd_merge    = None # 'corana_merge'
    cmd_procres  = None # 'corana_procres'
    batch_queue  = None # 'psnehq'
    pwdir        = None # echo $PWD

    fname_cfg    = None # 'ana-misc-exp/psana-xcsi0112-r0015-img-auto-correlation.cfg'
    fname_xtc    = None # '/reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc'
    fname_tau    = None # '/reg/d/psdm/XCS/xcsi0112/xtc/e167-r0015-s00-c00.xtc'

    dname        = None # '/reg/d/psdm/XCS/xcsi0112/xtc
    name         = None # 'e167-r0015-s00-c00.xtc'
    ext          = None # '.xtc'

    inst         = None # 'XCS'
    exp          = None # 'xcsi0112'
    run_str      = None # 'r0015'
    run_num      = None # 15

    fname_prefix = None # 'img-xcs'
    nfiles_out   = None # 8
    fname_com    = None # 'img-xcs-r0015'

    def __init__ ( self ) :
        print """__init__"""
        pass

    def set_default_pars(self, cmd_split, cmd_proc, cmd_merge, cmd_procres, batch_queue, pwdir) :
        self.cmd_split   = cmd_split
        self.cmd_proc    = cmd_proc
        self.cmd_merge   = cmd_merge
        self.cmd_procres = cmd_procres
        self.batch_queue = batch_queue
        self.pwdir       = pwdir

    def set_input_pars (self, fname_cfg, fname_xtc, fname_tau) :
        self.fname_cfg = fname_cfg
        self.fname_xtc = fname_xtc
        self.fname_tau = fname_tau

    def set_path_pars (self, dname, name, ext) :
        self.dname = dname
        self.name  = name
        self.ext   = ext

    def set_xtc_name_pars(self, inst, exp, run_str, run_num) :
        self.inst    = inst
        self.exp     = exp
        self.run_str = run_str
        self.run_num = run_num

    def set_cfg_file_pars(self, fname_prefix, nfiles_out, fname_com, dname_work) :
        self.fname_prefix = fname_prefix
        self.nfiles_out   = nfiles_out
        self.fname_com    = fname_com
        self.dname_work   = dname_work
        
    def print_pars ( self ) :
        print """CorAnaPars::print_pars()"""
        print 'fname config     :', self.fname_cfg
        print 'fname xtc        :', self.fname_xtc
        print 'fname tau        :', self.fname_tau
        print 'xtc file dirname :', self.dname
        print 'xtc file name    :', self.name
        print 'xtc file ext     :', self.ext
        print 'inst             :', self.inst   
        print 'exp              :', self.exp    
        print 'run_str          :', self.run_str
        print 'run_num          :', self.run_num
        print 'fname_prefix     :', self.fname_prefix
        print 'nfiles_out       :', self.nfiles_out
        print 'fname_com        :', self.fname_com
        print 'dname_work       :', self.dname_work
        print 'cmd_split        :', self.cmd_split  
        print 'cmd_proc         :', self.cmd_proc   
        print 'cmd_merge        :', self.cmd_merge  
        print 'cmd_procres      :', self.cmd_procres  
        print 'batch_queue      :', self.batch_queue
        print 'pwdir            :', self.pwdir
        print '                 :', 

#---------------------------------------
# Makes a single object of this class --
#---------------------------------------

coranapars = CorAnaPars()

#--------------------

if __name__ == '__main__' :

    coranapars.print_pars()

    sys.exit('The End')

#--------------------
