from buildbot.scheduler import Nightly, Dependent
from time import *

class LusiNightly ( Nightly ) :

    """ Scheduler class which sets additional property before starting build.
        The property names is 'bdate' and its value is the current date in the
        format "YYYYMMDD"
    """
    
    def doPeriodicBuild(self):
    
        t = localtime( time() )    
        self.properties.setProperty( 'bdate', strftime("%Y%m%d",t), "Scheduler" )
        
        # call base class to do real work
        Nightly.doPeriodicBuild ( self )

        
class LusiDependent ( Dependent ) :

    """ Scheduler which sets 'bdate' property from upstream scheduler """
    
    def upstreamBuilt(self, ss):

        bdate = self.upstream.properties.getProperty( 'bdate', None )
        if not bdate :
            t = localtime( time() )
            bdate = strftime("%Y%m%d",t)
        self.properties.setProperty( 'bdate', bdate, "Scheduler" )
        
        Dependent.upstreamBuilt(self, ss)
