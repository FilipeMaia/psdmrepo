#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: dump_simple.py 2622 2011-11-11 14:35:00Z salnikov@SLAC.STANFORD.EDU $
#
# Description:
#  Class DumpCsPad
#
# Author List:
#  Joseph S. Barrera III
#------------------------------------------------------------------------

"""Example module for accessing SharedIpimb data.

This software was developed for the LCLS project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@version $Id: dump_simple.py 2622 2011-11-11 14:35:00Z salnikov@SLAC.STANFORD.EDU $

@author Joseph S. Barrera III
"""

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
        self.m_src = source

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

        print "self.m_src = %s" % (self.m_src)

        self.source = env.configStr("source", "DetInfo(:Cspad)")

        print "configV1 =", env.getConfig("Psana::CsPad::ConfigV1", self.source)
        print "configV2 =", env.getConfig("Psana::CsPad::ConfigV2", self.source)
        print "configV3 =", env.getConfig("Psana::CsPad::ConfigV3", self.source)

        config = env.getConfig("Psana::CsPad::ConfigV3", self.source)
        if not config:
            config = env.getConfig("Psana::CsPad::ConfigV2", self.source)
        if not config:
            config = env.getConfig("Psana::CsPad::ConfigV1", self.source)
        if not config:
            print "unrecoginized CsPad::Config version"
            sys.exit(1)

        print "config type is ", type(config)


        version = config.concentratorVersion()
        print "version=", version
        s = "CsPad::ConfigV1:"
        s = s + "\n  concentratorVersion = " + str(config.concentratorVersion())
        s = s + "\n  runDelay = " + str(config.runDelay())
        s = s + "\n  eventCode = " + str(config.eventCode())
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
        """
        if hasattr(config, "protectionEnable"):
            s = s + "\n  protectionEnable = " + str(config.protectionEnable())
        if hasattr(config, "roiMask"):
            s = s + "\n  roiMask = " + str(config.roiMask())
        if hasattr(config, "numAsicsStored"):
            s = s + "\n  numAsicsStored = " + str(config.numAsicsStored())
        """
        print s

    def event(self, evt, env):
        evt_data = evt.getData("Psana::CsPad::DataV2", self.source)
        if not evt_data:
            evt_data = evt.getData("Psana::CsPad::DataV1", self.source)
            if not evt_data:
                print "data is of unexpected format"
                sys.exit(1)
        print evt_data

        s = "CsPad::DataV1:"
        print "evt_data = ", evt_data
        print "evt_data.quads_shape() = ", evt_data.quads_shape()
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

            # str << "\n    sb_temp = [ ";
            # const ndarray<uint16_t, 1>& sb_temp = el.sb_temp();
            # std::copy(sb_temp.begin(), sb_temp.end(), std::ostream_iterator<uint16_t>(str, " "));
            # str << "]";
            sb_temp = el.sb_temp()
            print "sb_temp = ", sb_temp
            print "type(sb_temp) = ", type(sb_temp)
            print "sb_temp.size = ", sb_temp.shape[0]
            s = s + "\n    sb_temp = [ "
            for i in range(sb_temp.shape[0]):
                s = s + str(sb_temp[i]) + " "
            s = s + "]"
            print s # XXX

            # const ndarray<int16_t, 3>& data = el.data();
            # str << "\n    data_shape = [ ";
            # std::copy(data.shape(), data.shape()+3, std::ostream_iterator<int>(str, " "));
            # str << "]";
            data = el.data()
            print "data.ndim = ", data.ndim
            s = s + "\n    data_shape = [ "
            for i in range(data.ndim):
                s = s + str(data.shape[i]) + " "
            s = s + "]"
            print s # XXX

            # str << "\n    common_mode = [ ";
            # for (unsigned i = 0; i != data.shape()[0]; ++ i) {
            #   str << el.common_mode(i) << ' ';
            # }
            # str << "]";
            s = s + "\n    common_mode = [ "
            for i in range(data.shape[0]):
                print "i=", i
                print "el.common_mode(i)=", el.common_mode(i)
                s = s + str(el.common_mode(i)) + " "
            s = s + "]"
            print s # XXX

            # str << "\n    data = [";
            # for (unsigned s = 0; s != data.shape()[0]; ++ s) {
            #   str << "\n        [ ";
            #   for (unsigned i = 0; i < 10; ++ i) str << data[s][0][i] << ' ';
            #   str << "... ]";
            # }
            # str << "\n    ]";
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
