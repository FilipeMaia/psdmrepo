#--------------------------------------------------------------------------
# File and Version Information:
# 	$Id:
#
# Description:
#	Class DumpPartition...
#
# Author List:
#
#------------------------------------------------------------------------

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
from psana import *

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _printPartition(config):
    print "Partition::ConfigV1:"
    print "  bldMask: 0x%16.16x" % config.bldMask()
    print "  numSources: %d" % config.numSources()
    sources = config.sources()
    for source in sources:
        src = source.src()
        group = source.group()
        print "    src= %s group= %s" % (src,group)
    
#---------------------
#  Class definition --
#---------------------
class dump_partition (object) :
    '''Class whose instance will be used as a user analysis module.'''

    #----------------
    #  Constructor --
    #----------------
    def __init__(self):
        '''Class constructor. Does not need any parameters'''
        pass

    #-------------------
    #  Public methods --
    #-------------------

    def beginJob( self, evt, env ) :
        """This optional method is called if present at the beginning 
        of the new run.

        @param evt    event data object
        @param env    environment object
        """

        config = env.configStore().get(Partition.ConfigV1, Source("ProcInfo()"))
        if config:
            print "dump_partition in beginJob()"
            _printPartition(config)


    def event( self, evt, env ) :
        """This method is called for every L1Accept transition.

        @param evt    event data object
        @param env    environment object
        """
        pass
