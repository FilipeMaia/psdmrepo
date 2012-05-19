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
from ExpNameDb.ExpNameDatabase import ExpNameDatabase

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
        self.m_expnum = 0

        if not files:
            logging.debug("ExpNameFromXtc: file list is missing, instrument/experiment will be empty")
            return
        
        expNums = set(io.XtcFileName(file).expNum() for file in files)
        if len(expNums) != 1: raise ValueError("ExpNameFromXtc: XTC files belong to different experiments: " + str(files))
        
        self.m_expnum = expNums.pop()
        logging.debug("ExpNameFromXtc: experiment number = %s", self.m_expnum)
        if self.m_expnum is None: return

        # find rest in experiment database
        expdb = ExpNameDatabase()
        self.m_instrument, self.m_experiment = expdb.getNames(self.m_expnum)
        if not self.m_instrument:
            logging.warning("ExpNameFromXtc: failed to find experiment ID = %d", self.m_expnum)

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
    
    def expNum(self):
        """
        self.expNum() -> int
        
        Returns experiment ID, or 0.
        """
        return self.m_expnum
    


class ExpNameFromConfig(object):
    """Class which determines experiment name from configuration string"""

    #----------------
    #  Constructor --
    #----------------
    def __init__(self, expname):
        """Constructor takes experiment name in the form XPP:xpp12311 or xpp12311."""

        self.m_instrument = ""
        self.m_experiment = ""
        self.m_expnum = 0

        words = expname.split(':')
        if len(words) == 1:
            self.m_experiment = words[0]
        elif len(words) == 2:
            self.m_instrument = words[0]
            self.m_experiment = words[1]
        else:
            raise ValueError("ExpNameFromConfig: experiment name has unexpected format: " + expname)

        # find rest in experiment database
        expdb = ExpNameDatabase()
        if not self.m_instrument:
            self.m_instrument, self.m_expnum = expdb.getInstrumentAndID(self.m_experiment)
            if not self.m_instrument:
                logging.warning("ExpNameFromConfig: failed to find experiment = %s", self.m_experiment)
        else:
            self.m_expnum = expdb.getID(self.m_instrument, self.m_experiment)
            if not self.m_expnum:
                logging.warning("ExpNameFromConfig: failed to find experiment = %s:%s", self.m_instrument, self.m_experiment)


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
    
    def expNum(self):
        """
        self.expNum() -> int
        
        Returns experiment ID, or 0.
        """
        return self.m_expnum


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
