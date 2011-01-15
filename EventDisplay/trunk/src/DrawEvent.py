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
import PrintHDF5        as printh5

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
        cp.confpars.h5_file_is_open = False
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

        if not cp.confpars.h5_file_is_open :
            self.openHDF5File()
    
        print 'Draw event %d' % ( cp.confpars.eventCurrent )
        runNumber = self.h5file.attrs['runNumber']
        #print 'Run number = %d' % (runNumber) 

        if runNumber < 600 :
            self.imageOfCSpadV1( mode )
        else :
            self.imageOfCSpadV2( mode )


    def imageOfCSpadV1 ( self, mode=1 ) :
        """Draws CSpad image for current event
        """
        dsname = "/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV1/XppGon.0:Cspad.0/data" # V1 for runs ~546,547...
        ds     = self.h5file[dsname]
        arr1ev = ds[cp.confpars.eventCurrent]
        self.plotsCSpad.plotCSpadV1(arr1ev, mode)
        #print 'Here should be an image of CSpad...'


    def imageOfCSpadV2 ( self, mode=1 ) :
        """Draws CSpad image for current event
        """
        dsname = "/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/XppGon.0:Cspad.0/data" # V2 for runs ~900
        ds     = self.h5file[dsname]
        arr1quad = ds[cp.confpars.eventCurrent]
        self.plotsCSpad.plotCSpadV2(arr1quad, mode)
        #print 'Here should be an image of CSpad...'


    def stopDrawEvent ( self ) :
        """Operations in case of stop drawing event(s)
        """
        print 'stopDrawEvent()'
        #self.plotsCSpad.close_fig1()
        #self.closeHDF5File()
        self.drawEvent() # mode=1 by default


    def quitDrawEvent ( self ) :
        """Operations in case of quit drawing event(s)
        """
        print 'quitDrawEvent()'
        self.plotsCSpad.close_fig1()
        self.closeHDF5File()
            

    def openHDF5File( self ) :     
        fname = cp.confpars.dirName+'/'+cp.confpars.fileName
        print 'openHDF5File() : %s' % (fname)
        self.h5file=  h5py.File(fname, 'r') # open read-only       
        cp.confpars.h5_file_is_open = True
        printh5.print_file_info(self.h5file)


    def closeHDF5File( self ) :       
        if cp.confpars.h5_file_is_open :
            self.h5file.close()
            cp.confpars.h5_file_is_open = False
            print 'closeHDF5File()'


#
#  In case someone decides to run this module
#
if __name__ == "__main__" :
    # In principle we can try to run test suite for this module,
    # have to think about it later. Right now just abort.
    sys.exit ( "Module is not supposed to be run as main module" )
