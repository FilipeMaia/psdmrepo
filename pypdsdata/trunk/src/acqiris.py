#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module acqiris...
#
#------------------------------------------------------------------------

"""Wrapper module for _pdsdata.acqiris, provides additional functionality.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
"""

#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
import numpy

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

# many classes are unchanged
from _pdsdata.acqiris import *

# extend DataDescV1
class DataDescV1(object) :
    
    def __init__(self, ddesc, hcfg, vcfg ):
        """ Constructor takes _pdsdata.acqiris.DataDescV1,
        _pdsdata.acqiris.HorizV1 and _pdsdata.acqiris.VertV1 objects """
        
        self.__ddesc = ddesc
        self.__hcfg = hcfg
        self.__vcfg = vcfg
    
    def nbrSamplesInSeg(self) :
        "Returns number of samples per segment"
        return self.__ddesc.nbrSamplesInSeg()
    
    def nbrSegments(self) :
        "Returns number of segments"
        return self.__ddesc.nbrSegments()
    
    def timestamp(self, segment) :
        "Returns TimestampV1 object for a given segment"
        return self.__ddesc.timestamp(segment)
 
    def waveform(self) :
        "Returns waveform array"
        wf = self.__ddesc.waveform( self.__hcfg )
        slope = self.__vcfg.slope()
        offset = self.__vcfg.offset()
        wf = wf * slope - offset
        return wf

    def timestamps(self) :
        "Returns array of timestamps"
        sampInterval = self.__hcfg.sampInterval()
        nbrSamples = self.__hcfg.nbrSamples()
        return numpy.arange( 0, nbrSamples*sampInterval, sampInterval, dtype=float )

