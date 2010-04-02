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
class myana ( object ) :
    """Example analysis class which reads waveform data and fill a 
    profile histogram with the waveforms """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, nenergy, e1, e2 ) :
        """Constructor. The parameters to the constructor are passed
        from pyana.cfg file. If parameters do not have default values
        here then the must be defined in pyana.cfg. All parameters are
        passed as strings, convert to correct type before use."""
        
        self.shotCountITof = 0

        self.nenergy = int(nenergy)
        self.e1 = float(e1)
        self.e2 = float(e2)
        logging.info( "nenergy=%d e1=%f e2=%f", self.nenergy, self.e1, self.e2 )

    #-------------------
    #  Public methods --
    #-------------------

    def beginjob( self, evt, env ) :
        """This method is called once at the beginning of the job"""

        logging.info( "myana.beginjob() called" )

        # open output files
        paramFileName = env.jobName()+".param"
        self.paramFile = env.mkfile(paramFileName)
        
        shotsFileName = paramFileName + '.data'
        self.shotsFile = env.mkfile(shotsFileName)
        print >> self.shotsFile, shotsFileName

        # get Acqiris configuration
        self.itofConfig = evt.getAcqConfig( "AmoITof" )
        if not self.itofConfig :  raise Exception("AmoITof configuration missing")
        
        self.etofConfig = evt.getAcqConfig( "AmoETof" )
        if not self.etofConfig :  raise Exception("AmoETof configuration missing")
        
        # book histograms
        hmgr = env.hmgr()

        self.itofHistos = []
        for i in range(self.nenergy) :
            
            name = "ITofavg%d" % i

            hconfig = self.itofConfig.horiz()
            halfbinsize = hconfig.sampInterval()/2.0
            
            prof = hmgr.prof( name, name, hconfig.nbrSamples(),
                                 0.0-halfbinsize, hconfig.sampInterval()*hconfig.nbrSamples()-halfbinsize)
            prof.SetYTitle("Volts")
            prof.SetXTitle("Seconds")
            self.itofHistos.append( prof )

        self.etofHistos = []
        for i in range(self.etofConfig.nbrChannels()) :
            
            name = "ETOF Channel %d" % i

            hconfig = self.etofConfig.horiz()
            prof = hmgr.prof( name, name, hconfig.nbrSamples(),
                                 0.0, hconfig.sampInterval()*hconfig.nbrSamples() )
            prof.SetYTitle("Volts")
            prof.SetXTitle("Seconds")
            self.etofHistos.append( prof )
        
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
        print >> self.paramFile, "%d %f %f %f %f" % (self.shotCountITof, 
                    shotEnergy[0], shotEnergy[1], shotEnergy[2], shotEnergy[3] )

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
        self.itofHistos[bin].FillN( len(ts), ts, wf, numpy.ones_like(ts) )

#        c = 1
#        for h in self.itofHistos :
#            self.canvas.cd(c)
#            h.Draw()
#            c += 1
#        self.canvas.Update()

    def endrun( self, env ) :
        """This method is called at the end of the run"""
        logging.info( "myana.endrun() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job, close your files"""
        logging.info( "myana.endjob() called" )

        self.paramFile.close()
        self.shotsFile.close()
