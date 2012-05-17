#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Class DumpCsPad
#
# Author List:
#  Joseph S. Barrera III
#------------------------------------------------------------------------

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 2622 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import logging

#-----------------------------
# Imports for other modules --
#-----------------------------
from pypdsdata import xtc

#----------------------------------
# Local non-exported definitions --
#----------------------------------

# local definitions usually start with _

#---------------------
#  Class definition --
#---------------------
class DumpCsPad(object) :
    """Class whose instance will be used as a user analysis module. """

    #----------------
    #  Constructor --
    #----------------
    def __init__ (self, source=""):
        pass

    #-------------------
    #  Public methods --
    #-------------------
    def beginCalibCycle(self, evt, env):

        print "in beginCalibCycle()"

        print "jobName=", env.jobName()
        print "instrument=", env.instrument()
        print "experiment=", env.experiment()
        print "expNum=", env.expNum()
        print "calibDir=", env.calibDir()

        self.source = env.configStr("source", "DetInfo(:Cspad)")
        config = env.configStore().get("Psana::CsPad::Config", self.source)

        s = type(config).__name__ + ":"
        s = s + "\n  concentratorVersion = " + str(config.concentratorVersion())
        s = s + "\n  runDelay = " + str(config.runDelay())
        s = s + "\n  eventCode = " + str(config.eventCode())

        if hasattr(config, "protectionEnable"):
            s = s + "\n  protectionEnable = " + str(config.protectionEnable())

        if hasattr(config, "protectionThresholds"):
            s = s + "\n  protectionThresholds:"
            for i in range(0, config.numQuads()):
                s = s + "\n    adcThreshold = " + str(config.protectionThresholds(i).adcThreshold())
                s = s + " pixelCountThreshold = " + str(config.protectionThresholds(i).pixelCountThreshold())

        s = s + "\n  inactiveRunMode = " + str(config.inactiveRunMode())
        s = s + "\n  activeRunMode = " + str(config.activeRunMode())
        s = s + "\n  tdi = " + str(config.tdi())
        s = s + "\n  payloadSize = " + str(config.payloadSize())
        s = s + "\n  badAsicMask0 = " + str(config.badAsicMask0())
        s = s + "\n  badAsicMask1 = " + str(config.badAsicMask1())
        s = s + "\n  asicMask = " + str(config.asicMask())
        s = s + "\n  quadMask = " + str(config.quadMask())
        s = s + "\n  numAsicsRead = " + str(config.numAsicsRead())
        s = s + "\n  numQuads = " + str(config.numQuads())
        s = s + "\n  numSect = " + str(config.numSect())

        if hasattr(config, "roiMask"):
            s = s + "\n  roiMask = ???"

        if hasattr(config, "numAsicsStored"):
            s = s + "\n  numAsicsStored = ???"

        print s

    def event(self, evt, env):
        evt_data = evt.get("Psana::CsPad::Data", self.source)
        print evt_data
        print type(evt_data)

        s = type(evt_data).__name__ + ":"
        nQuads = evt_data.quads_shape()[0]
        for q in range(nQuads):
            el = evt_data.quads(q)

            s = s + "\n  Element #" + str(q)
            s = s + "\n    virtual_channel = " + str(el.virtual_channel())
            s = s + "\n    lane = " + str(el.lane())
            s = s + "\n    tid = " + str(el.tid())
            s = s + "\n    acq_count = " + str(el.acq_count())
            s = s + "\n    op_code = " + str(el.op_code())
            s = s + "\n    quad = " + str(el.quad())
            s = s + "\n    seq_count = " + str(el.seq_count())
            s = s + "\n    ticks = " + str(el.ticks())
            s = s + "\n    fiducials = " + str(el.fiducials())
            s = s + "\n    frame_type = " + str(el.frame_type())

            sb_temp = el.sb_temp()
            print "sb_temp = ", sb_temp
            print "type(sb_temp) = ", type(sb_temp)
            print "sb_temp.size = ", sb_temp.shape[0]
            s = s + "\n    sb_temp = [ "
            for i in range(sb_temp.shape[0]):
                s = s + str(sb_temp[i]) + " "
            s = s + "]"

            data = el.data()
            print "data.ndim = ", data.ndim
            s = s + "\n    data_shape = [ "
            for i in range(data.ndim):
                s = s + str(data.shape[i]) + " "
            s = s + "]"

            s = s + "\n    common_mode = [ "
            for i in range(data.shape[0]):
                print "i=", i
                print "el.common_mode(i)=", el.common_mode(i)
                s = s + str(el.common_mode(i)) + " "
            s = s + "]"

            s = s + "\n    data = ["
            for i in range(data.shape[0]):
                s = s + "\n        [ "
                for j in range(10):
                    s = s + str(data[i][0][j]) + " "
                s = s + "... ]"
            s = s + "\n    ]"
            print s

    def endJob(self, evt, env):
        pass
