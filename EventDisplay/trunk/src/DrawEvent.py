#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module DrawEvent...
#
#------------------------------------------------------------------------

"""Reads info from HDF5 file and rendering it depending on configuration parameters.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

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
import h5py    # access to hdf5 file structure
from numpy import *  # for use like       array(...)

#-----------------------------
# Imports for other modules --
#-----------------------------
import ConfigParameters as cp
import PlotsForCSpad    as cspad

#---------------------
#  Class definition --
#---------------------
class DrawEvent ( object ) :
    """Reads info from HDF5 file and rendering it depending on configuration parameters.

    @see BaseClass
    @see OtherClass
    """

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor.
        """
        print 'DrawEvent () Initialization'
        self.h5_file_is_open = False
        self.plotsCSpad      = cspad.PlotsForCSpad()

    #-------------------
    #  Public methods --
    #-------------------

    def drawEvent ( self, mode=1 ) :
        """Draws current event

        Explanation of what it does.
        @param x   first parameter
        @param y   second parameter
        @return    return value
        """

        if not self.h5_file_is_open :
            self.openHDF5File()
    
        print 'Draw event %d' % ( cp.confpars.eventCurrent )
        self.imageOfCSpad( mode )


    def imageOfCSpad ( self, mode=1 ) :
        """Draws CSpad image for current event
        """
        dsname = "/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV1/XppGon.0:Cspad.0/data"
        ds     = self.h5file[dsname]
        arr1ev = ds[cp.confpars.eventCurrent]
        self.plotsCSpad.plot_CSpad(arr1ev, mode)
        #print 'Here should be an image of CSpad...'


    def stopDrawEvent ( self ) :
        """Operations in case of stop drawing event(s)
        """
        print 'stopDrawEvent()'
        self.plotsCSpad.close_fig1()
        self.closeHDF5File()


    def openHDF5File( self ) :     
        fname = cp.confpars.dirName+'/'+cp.confpars.fileName
        print 'openHDF5File() : %s' % (fname)
        self.h5file = h5py.File(fname, 'r') # open read-only       
        self.h5_file_is_open = True


    def closeHDF5File( self ) :       
        self.h5file.close()
        print 'closeHDF5File()'
        self.h5_file_is_open = False


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
