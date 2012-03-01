#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module expname...
#
#------------------------------------------------------------------------

"""Classes which provide access to instrument/experiment name in pyana.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id$

@author Andy Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import io
from AppUtils.AppDataPath import AppDataPath

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class ExpNameFromXtc(object):
    """Class which determines experiment name from XTC file name"""

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, files):
        """Constructor takes a list of file names."""

        self.m_instrument = ""
        self.m_experiment = ""

        if not files:
            logging.debug("ExpNameFromXtc: file list is missing, instrument/experiment will be empty")
            return
        
        expNums = set(io.XtcFileName(file).expNum() for file in files)
        if len(expNums) != 1: raise ValueError("ExpNameFromXtc: XTC files belong to different experiments: " + str(files))
        
        expNum = expNums.pop()
        logging.debug("ExpNameFromXtc: experiment number = %s", expNum)
        if expNum is None: return

        # find/open experiment database
        expdbpath = AppDataPath("psana/experiment-db.dat")
        expdbpath = expdbpath.path()
        logging.debug("ExpNameFromXtc: experiment database = %s", expdbpath)
        if not expdbpath: return
        
        # read database
        for line in open(expdbpath):
            num, instr, exper = line.split()
            if int(num) == expNum:
                self.m_instrument = instr
                self.m_experiment = exper
                logging.debug("ExpNameFromXtc: found num=%s instr=%s exper=%s", num, instr, exper)
                break

    #-------------------
    #  Public methods --
    #-------------------

    def instrument(self):
        """
        self.instrument() -> string
        
        Returns instrument name, or empty string if name is not known
        """
        return self.m_instrument
    
    def experiment(self):
        """
        self.experiment() -> string
        
        Returns experiment name, or empty string if name is not known
        """
        return self.m_experiment
    


class ExpNameFromConfig(object):
    """Class which determines experiment name from configuration string"""

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, expname):
        """Constructor takes experiment name in the form XPP:xpp12311 or xpp12311."""

        self.m_instrument = ""
        self.m_experiment = ""

        words = expname.split(':')
        if len(words) == 1:
            self.m_instrument = words[0][:3].upper()
            self.m_experiment = words[0]
        elif len(words) == 2:
            self.m_instrument = words[0]
            self.m_experiment = words[1]
        else:
            raise ValueError("ExpNameFromConfig: experiment name has unexpected format: " + expname)

        logging.debug("ExpNameFromConfig: instr=%s exper=%s", num, instr, exper)

    #-------------------
    #  Public methods --
    #-------------------

    def instrument(self):
        """
        self.instrument() -> string
        
        Returns instrument name, or empty string if name is not known
        """
        return self.m_instrument
    
    def experiment(self):
        """
        self.experiment() -> string
        
        Returns experiment name, or empty string if name is not known
        """
        return self.m_experiment
    


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
