from twisted.python import log
from zope.interface import implements
from buildbot.scheduler import Nightly, Dependent, BaseUpstreamScheduler
from buildbot import interfaces, buildset
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


class MultiDependent(BaseUpstreamScheduler):
    """This scheduler runs some set of 'downstream' builds when one or
    more 'upstream' schedulers have completed successfully."""

    implements(interfaces.IDownstreamScheduler)

    compare_attrs = ('name', 'upstream', 'builderNames', 'properties')

    def __init__(self, name, upstreams, builderNames, properties={}):
        for upstream in upstreams:
            assert interfaces.IUpstreamScheduler.providedBy(upstream)
        assert len(upstreams)
        BaseUpstreamScheduler.__init__(self, name, properties)
        self.upstream_names = [upstream.name for upstream in upstreams]
        self.upstreams = None
        self.builderNames = builderNames
        self.count = 0
        log.msg("MultiDependent: self.upstream_names = %s" % self.upstream_names)

    def listBuilderNames(self):
        return self.builderNames

    def getPendingBuildTimes(self):
        # report the upstream's value
        return self.findUpstreamScheduler(self.upstream_names[0]).getPendingBuildTimes()

    def startService(self):
        BaseUpstreamScheduler.startService(self)
        self.upstreams = map(self.findUpstreamScheduler, self.upstream_names)
        self.upstreams[0].subscribeToSuccessfulBuilds(self.rootUpstreamBuilt)
        for upstream in self.upstreams[1:]:
            upstream.subscribeToSuccessfulBuilds(self.upstreamBuilt)
        self.count = 0

    def stopService(self):
        d = BaseUpstreamScheduler.stopService(self)
        self.upstreams[0].unsubscribeToSuccessfulBuilds(self.rootUpstreamBuilt)
        for upstream in self.upstreams[1:]:
            upstream.unsubscribeToSuccessfulBuilds(self.upstreamBuilt)
        self.upstreams = None
        self.count = 0
        return d

    def rootUpstreamBuilt(self, ss):
        self.count = 1
        log.msg("MultiDependent: root upstream finished, self.count = %d" % self.count)
        if self.count == len(self.upstreams):
            bs = buildset.BuildSet(self.builderNames, ss, properties=self.properties)
            self.submitBuildSet(bs)

    def upstreamBuilt(self, ss):
        self.count += 1
        log.msg("MultiDependent: non-root upstream finished, self.count = %d" % self.count)
        if self.count == len(self.upstreams):
            bs = buildset.BuildSet(self.builderNames, ss, properties=self.properties)
            self.submitBuildSet(bs)

    def findUpstreamScheduler(self, name):
        # find our *active* upstream scheduler (which may not be self.upstream!) by name
        upstream = None
        for s in self.parent.allSchedulers():
            if s.name == name and interfaces.IUpstreamScheduler.providedBy(s):
                upstream = s
        if not upstream:
            log.msg("ERROR: Couldn't find upstream scheduler of name <%s>" % name)
        return upstream

    def checkUpstreamScheduler(self):
        # if we don't already have an upstream, then there's nothing to worry about
        if not self.upstreams:
            return

        upstreams = map(self.findUpstreamScheduler, self.upstream_names)

        # if it's already correct, we're good to go
        if upstreams == self.upstreams:
            return

        # to avoid confusion unsubscribe from old upstreams and subscribe to new ones
        self.upstreams[0].unsubscribeToSuccessfulBuilds(self.rootUpstreamBuilt)
        for upstream in self.upstreams[1:]:
            upstream.unsubscribeToSuccessfulBuilds(self.upstreamBuilt)
        self.upstreams = upstreams
        self.upstreams[0].subscribeToSuccessfulBuilds(self.rootUpstreamBuilt)
        for upstream in self.upstreams[1:]:
            upstream.subscribeToSuccessfulBuilds(self.upstreamBuilt)

        log.msg("Dependent <%s> connected to new Upstreams %s" %
                (self.name, self.upstream_names))
