#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: myana1.py 1093 2010-07-07 22:49:25Z salnikov $
#
# Description:
#  Module myana...
#
#------------------------------------------------------------------------

"""User analysis job for XTC data.

This software was developed for the LUSI project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: myana1.py 1093 2010-07-07 22:49:25Z salnikov $

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision: 1093 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging
import numpy

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class myana1 ( object ) :
    """Example analysis job that reads waveform and does something useless"""

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, nenergy, e1, e2 ) :
        """Constructor."""
        
        self.nenergy = int(nenergy)
        self.e1 = float(e1)
        self.e2 = float(e2)
        logging.info( "nenergy=%d e1=%f e2=%f", self.nenergy, self.e1, self.e2 )
        self.shotCountITof = 0
        self.sum2 = None

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job"""

        logging.info( "myana1.beginjob() called" )

    def beginrun( self, evt, env ) :
        """This method is called at the beginning of the run"""

        logging.info( "myana1.beginrun() called" )

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition"""

        # store energy in a file
        shotEnergy = evt.getFeeGasDet()
        if not shotEnergy : shotEnergy = [0,0,0,0]

        # determine energy bin 
        energy=(shotEnergy[2]+shotEnergy[3])/2-self.e1
        bin = int((energy/(self.e2-self.e1))*self.nenergy+0.5)
        if bin<1 : bin=0
        if bin>self.nenergy-1 : bin=self.nenergy-1
        
        # get Acqiris data
        channel = 0
        ddesc = evt.getAcqValue( "AmoITof", channel, env )
        if ddesc:
            wf = ddesc.waveform()
            ts = ddesc.timestamps()
            if self.sum2 is None :
                self.sum2 = wf**2
            else :
                self.sum2 += wf**2

    def endrun( self, env ) :
        """This method is called at the end of the run"""
        logging.info( "myana1.endrun() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job, close your files"""
        logging.info( "myana1.endjob() called" )
