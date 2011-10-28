
#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module CalibCycles...
#
#------------------------------------------------------------------------

"""Helper for operations with CalibCycles.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id:$

@author Mikhail S. Dubrovin
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 4 $"
# $Source$

#-------------------------------
# Imports of standard modules --
#-------------------------------
import sys
import os
import h5py

#-----------------------------
# Imports for other modules --
#-----------------------------
import GlobalMethods as gm

#-----------------------------
#  Class definition --
#-----------------------------
class CalibCycles :
    """Helper class for operations with CalibCycles
    """

    def __init__ ( self, h5fname=None, dsname=None ) :
        """Initialization

        if the h5fname is missing, than the self.numberOfCalibCycles
        should be defined from open file by extractNumberOfCalibCyclesFromOpenFile(...)
        """

        #self.h5fname = h5fname
        #self.dsname  = dsname
        if h5fname != None :
            self.numberOfCalibCycles = self.extractNumberOfCalibCycles(h5fname, dsname)
        #else :
        #    print 'WARNING: THE CalibCycles INITIALIZATION IS OMITTED...'
            

#-----------------------------

    def openHDF5File(self, h5fname) :
        self.h5file = h5py.File(h5fname, 'r')
        return self.h5file

#-----------------------------

    def closeHDF5File(self) :
        self.h5file.close()

#-----------------------------

    def extractNumberOfCalibCycles(self, h5fname, dsname) :
        h5file = self.openHDF5File(h5fname)
        numberOfCalibCycles = self.extractNumberOfCalibCyclesFromOpenFile(h5file, dsname)
        self.closeHDF5File()
        return numberOfCalibCycles 

#-----------------------------

    def extractNumberOfCalibCyclesFromOpenFile(self, h5file, dsname) :
        runGroupName = self.getRunGroupName(dsname)
        g = h5file[runGroupName]
        self.numberOfCalibCycles = len(g.items())
        return self.numberOfCalibCycles 

#-----------------------------

    def getNumberOfCalibCycles(self) :
        return self.numberOfCalibCycles 

#-----------------------------

    def getRunGroupName(self, dsname) :
        s0, sN, isFoundInString = gm.getPatternEndsInTheString(dsname, pattern='Run:')
        if isFoundInString :
            runGroupName = dsname[0:sN+4]
            return runGroupName
        else :
            return None
        
#-----------------------------

    def getCalibCycleNumber(self, dsname) :
        s0, sN, isFoundInString = gm.getPatternEndsInTheString(dsname, pattern='CalibCycle:')
        if isFoundInString :
            calibcycleNumber = dsname[sN:sN+4]
            return int(calibcycleNumber)
        else :
            return None

#-----------------------------

    def get4DigitStringFromNumber(self, N) :
            str_number = '%04d' % N
            return str_number

#-----------------------------

    def getDSNameForCalibCycleNumber(self, dsname, Ncc) :
        s0, sN, isFoundInString = gm.getPatternEndsInTheString(dsname, pattern='CalibCycle:')
        if isFoundInString :
            #print 'dsname[sN:sN+4] = ', dsname[sN:sN+4]
            NccNew = self.checkNccLimits(Ncc)
            return dsname[0:sN] + self.get4DigitStringFromNumber(NccNew) + dsname[sN+4:]
        else :
            return None

#-----------------------------

    def checkNccLimits(self, Ncc) :
        NccNew = Ncc
        if  NccNew < 0 :
            NccNew = 0
            print 'WARNING: Requested CalibCycle number < 0: set NccNew =', NccNew
        if  NccNew >= self.numberOfCalibCycles :
            NccNew  = self.numberOfCalibCycles - 1
            print 'WARNING: Requested CalibCycle number > max: set NccNew =', NccNew
        return NccNew
    
#-----------------------------

    def getDSNameForCalibCycleDN(self, dsname, dNcc) :
        Ncc = self.getCalibCycleNumber(dsname)
        return self.getDSNameForCalibCycleNumber(dsname, Ncc + dNcc)

#-----------------------------
#  Test
#
if __name__ == "__main__" :

    h5fname = '/reg/d/psdm/AMO/amo30211/hdf5/amo30211-r0326.h5'
    dsname  = '/Configure:0000/Run:0000/CalibCycle:0012/Acqiris::DataDescV1/AmoETOF.0:Acqiris.0/waveforms'

    print 'h5fname                      =', h5fname
    print 'dsname                       =', dsname

    o = CalibCycles( h5fname, dsname )

    print 'Run group name               =', o.getRunGroupName(dsname)
    print 'CalibCycle number            =', o.getCalibCycleNumber(dsname)
    print 'Number of calibcycles        =', o.extractNumberOfCalibCycles(h5fname, dsname)
    print 'Number of calibcycles        =', o.getNumberOfCalibCycles()
    print '4-digit string for N=56      =', o.get4DigitStringFromNumber(56)
    print 'dsname for CalibCycle  17    =', o.getDSNameForCalibCycleNumber(dsname,17)
    print 'dsname for CalibCycle -11    =', o.getDSNameForCalibCycleNumber(dsname,-11)
    print 'dsname for CalibCycle  57    =', o.getDSNameForCalibCycleNumber(dsname,57)
    print 'dsname for CalibCycle N+1    =', o.getDSNameForCalibCycleDN(dsname, 1)
    print 'dsname for CalibCycle N-3    =', o.getDSNameForCalibCycleDN(dsname,-3)
    print 'dsname for CalibCycle N+2    =', o.getDSNameForCalibCycleDN(dsname, 2)
    print 'dsname for CalibCycle N+40   =', o.getDSNameForCalibCycleDN(dsname, 40)
    print 'dsname for CalibCycle N-40   =', o.getDSNameForCalibCycleDN(dsname,-40)

    sys.exit ( "End of test" )

#-----------------------------
