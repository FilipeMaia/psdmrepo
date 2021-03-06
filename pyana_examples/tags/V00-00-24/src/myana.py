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
from scipy import integrate

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
class myana ( object ) :
    """Example analysis class which reads waveform data and fill a 
    profile histogram with the waveforms """

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, name, nenergy, e1, e2 ) :
        """Constructor. The parameters to the constructor are passed
        from pyana.cfg file. If parameters do not have default values
        here then the must be defined in pyana.cfg. All parameters are
        passed as strings, convert to correct type before use."""
        
        self.shotCountITof = 0

        self.name = name
        self.nenergy = int(nenergy)
        self.e1 = float(e1)
        self.e2 = float(e2)
        logging.info( "name=%s nenergy=%d e1=%f e2=%f", self.name, self.nenergy, self.e1, self.e2 )

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
        self.itofConfig = env.getAcqConfig( "AmoITof" )
        if not self.itofConfig :  raise Exception("AmoITof configuration missing")
        
        self.etofConfig = env.getAcqConfig( "AmoETof" )
        if not self.etofConfig :  raise Exception("AmoETof configuration missing")
        
        # book histograms
        hmgr = env.hmgr()

        self.itofHistos = []
        self.itofHistosIntegral = []
        for i in range(self.nenergy) :
            
            name = "%s-ITofavg%d" % (self.name, i)

            hconfig = self.itofConfig.horiz()
            halfbinsize = hconfig.sampInterval()/2.0
            
            prof = hmgr.prof( name, name, hconfig.nbrSamples(),
                                 0.0-halfbinsize, hconfig.sampInterval()*hconfig.nbrSamples()-halfbinsize)
            prof.SetYTitle("Volts")
            prof.SetXTitle("Seconds")
            self.itofHistos.append( prof )

            name = "%s-ITofavgInteg%d" % (self.name, i)
            ihist = hmgr.h1i (name, name, 100, 2e-8, 6e-8 )

            self.itofHistosIntegral.append ( ihist )

        self.etofHistos = []
        for i in range(self.etofConfig.nbrChannels()) :
            
            name = "%s - ETOF Channel %d" % (self.name, i)

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
        """This optional method is called at the beginning of the run"""

        logging.info( "myana.beginrun() called" )

    def begincalibcycle( self, evt, env ) :
        """This optional method is called at the beginning of the calib cycle"""

        logging.info( "myana.begincalibcycle() called" )

    def event( self, evt, env ) :
        """This method is called for every L1Accept transition"""

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
        if ddesc :
            wf = ddesc.waveform()
            ts = ddesc.timestamps()
            self.itofHistos[bin].FillN( len(ts), ts, wf, numpy.ones_like(ts) )

            # integrate it
            integ = integrate.trapz (wf, ts)
            self.itofHistosIntegral[bin].Fill(integ)

    def endrun( self, env ) :
        """This optional method is called if present at the end of the run"""
        logging.info( "myana.endrun() called" )

    def endcalibcycle( self, env ) :
        """This optional method is called if present at the end of the calib cycle"""
        logging.info( "myana.endcalibcycle() called" )

    def endjob( self, env ) :
        """This method is called at the end of the job, close your files"""
        logging.info( "myana.endjob() called" )

        self.paramFile.close()
        self.shotsFile.close()
