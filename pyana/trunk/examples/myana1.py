#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module myana...
#
#------------------------------------------------------------------------

"""User analysis job for XTC data.

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
import sys
import logging
import numpy

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------
from pyana.histo import HistoMgr

from ROOT import TCanvas

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
    """Brief description of a class.

    Full description of this class. The whole purpose of this class is 
    to serve as an example for LUSI users. It shows the structure of
    the code inside the class. Class can have class (static) variables, 
    which can be private or public. It is good idea to define constructor 
    for your class (in Python there is only one constructor). Put your 
    public methods after constructor, and private methods after public.
    """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self ) :
        """Constructor."""
        
        self.shotCountITof = 0
        self.sum2 = None

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job"""

        logging.info( "myana.beginjob() called" )

        # get ITof parameters from myana.inp
        inpfile = open("myana.inp")
        self.nenergy = int(inpfile.readline())
        self.e1 = float(inpfile.readline())
        self.e2 = float(inpfile.readline())
        print "%d %f %f" % (self.nenergy, self.e1, self.e2)
        del inpfile

        
#        self.canvas = TCanvas('canvas', 'ITofavg histograms', 1000, 800)
#        self.canvas.Divide( 5, (len(self.itofHistos)+4)/5 )
#        c = 1
#        for h in self.itofHistos :
#            self.canvas.cd(c)
#            h.Draw()
#            c += 1
        

    def beginrun( self, evt, env ) :
        """This method is called at the beginning of the run"""

        logging.info( "myana.beginrun() called" )

    def event( self, evt, env ) :
        """This methiod is called for every L1Accept transition"""

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
        wf = ddesc.waveform()
        ts = ddesc.timestamps()
        if self.sum2 is None :
            self.sum2 = wf**2
        else :
            self.sum2 += wf**2

    def endrun( self, env ) :
        """This method is called at the end of the run"""
        logging.info( "myana.endrun() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job, close your files"""
        logging.info( "myana.endjob() called" )
