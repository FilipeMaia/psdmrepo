#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module ConfigParameters...
#
#------------------------------------------------------------------------

"""Is intended as a storage for configuration parameters.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: template!python!py 4 2008-10-08 19:27:36Z salnikov $

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os

#-----------------------------
# Imports for other modules --
#-----------------------------
#import ConfigParameters as cpbase
from ConfigParameters import * # ConfigParameters

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#---------------------
#  Class definition --
#---------------------

class ConfigParametersCorAna ( ConfigParameters ) :
    """Is intended as a storage for configuration parameters.
    #@see BaseClass ConfigParameters
    #@see OtherClass Parameters
    """

    list_pars = []

    def __init__ ( self, fname=None ) :
        """Constructor.
        @param fname  the file name with configuration parameters, if not specified then it will be set to the default value at declaration.
        """
        ConfigParameters.__init__(self)
        self.declareCorAnaParameters()
        self.readParametersFromFile ( fname )
        self.initRunTimeParameters()


    def initRunTimeParameters( self ) :
        #self.guiConfigParsIsOpen = False
        pass

    def declareCorAnaParameters( self ) :
        # Possible typs for declaration : 'str', 'int', 'long', 'float', 'bool' 
#       self.fname_cp           = self.declareParameter( name='FNAME_CONFIG_PARS', val_def='confpars.txt', typ='str' ) 
        self.fname_ped          = self.declareParameter( name='FNAME_PEDESTALS',   val_def='my_ped.txt',   typ='str' ) 
        self.fname_dat          = self.declareParameter( name='FNAME_DATA',        val_def='my_dat.txt',   typ='str' ) 
        self.instr_dir          = self.declareParameter( name='INSTRUMENT_DIR',    val_def='/reg/d/psdm',  typ='str' ) 
        self.instr_name         = self.declareParameter( name='INSTRUMENT_NAME',   val_def='XCS',          typ='str' ) 
        self.exp_name           = self.declareParameter( name='EXPERIMENT_NAME',   val_def='xcsi0112',     typ='str' ) 
        self.str_run_number     = self.declareParameter( name='RUN_NUMBER',        val_def='0015',         typ='str' ) 
        self.str_run_number_dark= self.declareParameter( name='RUN_NUMBER_DARK',   val_def='0014',         typ='str' ) 

#---------------------------------------

    def printParsDirectly( self ) :
        print 'Direct use of parameter:' + self.fname_cp .name(), self.fname_cp .value()  
        print 'Direct use of parameter:' + self.fname_ped.name(), self.fname_ped.value()      
        print 'Direct use of parameter:' + self.fname_dat.name(), self.fname_dat.value()      

#-----------------------------

confpars = ConfigParametersCorAna (fname=getConfigFileFromInput())

#-----------------------------
#
#  In case someone decides to run this module
#
if __name__ == "__main__" :

    confpars.printParameters()
    confpars.printParsDirectly()
    #confpars.saveParametersInFile()

    sys.exit ( 'End of test for ConfigParametersCorAna' )

#-----------------------------
