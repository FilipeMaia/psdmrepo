//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpCsPad...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpCsPad.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <iomanip>
#include <iterator>
#include <numeric>
#include <functional>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/cspad.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpCsPad)

namespace {

  void dumpQuadReg(std::ostream& str, const Psana::CsPad::ConfigV1QuadReg& quad)
  {
    str << "\n    shiftSelect = " << quad.shiftSelect();
    str << "\n    edgeSelect = " << quad.edgeSelect();
    str << "\n    readClkSet = " << quad.readClkSet();
    str << "\n    readClkHold = " << quad.readClkHold();
    str << "\n    dataMode = " << quad.dataMode();
    str << "\n    prstSel = " << quad.prstSel();
    str << "\n    acqDelay = " << quad.acqDelay();
    str << "\n    intTime = " << quad.intTime();
    str << "\n    digDelay = " << quad.digDelay();
    str << "\n    ampIdle = " << quad.ampIdle();
    str << "\n    injTotal = " << quad.injTotal();
    str << "\n    rowColShiftPer = " << quad.rowColShiftPer();
    str << "\n    digitalPots = " << quad.dp().pots();
    str << "\n    readOnly = shiftTest: " << quad.ro().shiftTest() << " verstion: " << quad.ro().version();
    str << "\n    gainMap = " << quad.gm().gainMap();
  }

  void dumpQuadReg(std::ostream& str, const Psana::CsPad::ConfigV2QuadReg& quad)
  {
    str << "\n    shiftSelect = " << quad.shiftSelect();
    str << "\n    edgeSelect = " << quad.edgeSelect();
    str << "\n    readClkSet = " << quad.readClkSet();
    str << "\n    readClkHold = " << quad.readClkHold();
    str << "\n    dataMode = " << quad.dataMode();
    str << "\n    prstSel = " << quad.prstSel();
    str << "\n    acqDelay = " << quad.acqDelay();
    str << "\n    intTime = " << quad.intTime();
    str << "\n    digDelay = " << quad.digDelay();
    str << "\n    ampIdle = " << quad.ampIdle();
    str << "\n    injTotal = " << quad.injTotal();
    str << "\n    rowColShiftPer = " << quad.rowColShiftPer();
    str << "\n    ampReset = " << quad.ampReset();
    str << "\n    digCount = " << quad.digCount();
    str << "\n    digPeriod = " << quad.digPeriod();
    str << "\n    digitalPots = " << quad.dp().pots();
    str << "\n    readOnly = shiftTest: " << quad.ro().shiftTest() << " verstion: " << quad.ro().version();
    str << "\n    gainMap = " << quad.gm().gainMap();
  }

  void dumpQuadReg(std::ostream& str, const Psana::CsPad::ConfigV3QuadReg& quad)
  {
    str << "\n    shiftSelect = " << quad.shiftSelect();
    str << "\n    edgeSelect = " << quad.edgeSelect();
    str << "\n    readClkSet = " << quad.readClkSet();
    str << "\n    readClkHold = " << quad.readClkHold();
    str << "\n    dataMode = " << quad.dataMode();
    str << "\n    prstSel = " << quad.prstSel();
    str << "\n    acqDelay = " << quad.acqDelay();
    str << "\n    intTime = " << quad.intTime();
    str << "\n    digDelay = " << quad.digDelay();
    str << "\n    ampIdle = " << quad.ampIdle();
    str << "\n    injTotal = " << quad.injTotal();
    str << "\n    rowColShiftPer = " << quad.rowColShiftPer();
    str << "\n    ampReset = " << quad.ampReset();
    str << "\n    digCount = " << quad.digCount();
    str << "\n    digPeriod = " << quad.digPeriod();
    str << "\n    biasTuning = " << quad.biasTuning();
    str << "\n    pdpmndnmBalance = " << quad.pdpmndnmBalance();
    str << "\n    digitalPots = " << quad.dp().pots();
    str << "\n    readOnly = shiftTest: " << quad.ro().shiftTest() << " verstion: " << quad.ro().version();
    str << "\n    gainMap = " << quad.gm().gainMap();
  }

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpCsPad::DumpCsPad (const std::string& name)
  : Module(name)
{
  m_key = configStr("inputKey", "");
  m_src = configSrc("source", "DetInfo(:Cspad)");
}

//--------------
// Destructor --
//--------------
DumpCsPad::~DumpCsPad ()
{
}

// Method which is called at the beginning of the calibration cycle
void 
DumpCsPad::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::CsPad::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {
    
    WithMsgLog(name(), info, str) {
      str << "CsPad::ConfigV1:";
      str << "\n  concentratorVersion = " << config1->concentratorVersion();
      str << "\n  runDelay = " << config1->runDelay();
      str << "\n  eventCode = " << config1->eventCode();
      str << "\n  inactiveRunMode = " << config1->inactiveRunMode();
      str << "\n  activeRunMode = " << config1->activeRunMode();
      str << "\n  tdi = " << config1->tdi();
      str << "\n  payloadSize = " << config1->payloadSize();
      str << "\n  badAsicMask0 = " << config1->badAsicMask0();
      str << "\n  badAsicMask1 = " << config1->badAsicMask1();
      str << "\n  asicMask = " << config1->asicMask();
      str << "\n  quadMask = " << config1->quadMask();
      str << "\n  numAsicsRead = " << config1->numAsicsRead();
      str << "\n  numQuads = " << config1->numQuads();
      str << "\n  numSect = " << config1->numSect();
      for (unsigned iq = 0; iq != config1->numQuads(); ++ iq) {
        str << "\n  quad #" << iq;
        dumpQuadReg(str, config1->quads(iq));
      }
    }
    
  }

  shared_ptr<Psana::CsPad::ConfigV2> config2 = env.configStore().get(m_src);
  if (config2) {
    
    WithMsgLog(name(), info, str) {
      str << "CsPad::ConfigV2:";
      str << "\n  concentratorVersion = " << config2->concentratorVersion();
      str << "\n  runDelay = " << config2->runDelay();
      str << "\n  eventCode = " << config2->eventCode();
      str << "\n  inactiveRunMode = " << config2->inactiveRunMode();
      str << "\n  activeRunMode = " << config2->activeRunMode();
      str << "\n  tdi = " << config2->tdi();
      str << "\n  payloadSize = " << config2->payloadSize();
      str << "\n  badAsicMask0 = " << config2->badAsicMask0();
      str << "\n  badAsicMask1 = " << config2->badAsicMask1();
      str << "\n  asicMask = " << config2->asicMask();
      str << "\n  quadMask = " << config2->quadMask();
      str << "\n  numAsicsRead = " << config2->numAsicsRead();
      str << "\n  numQuads = " << config2->numQuads();
      str << "\n  numSect = " << config2->numSect();
      str << "\n  roiMask =";
      for (unsigned i = 0; i < config2->numQuads(); ++ i) {
        str.setf(std::ios::showbase);
        str << ' ' << std::hex << config2->roiMask(i) << std::dec;
      }
      str << "\n  numAsicsStored =";
      for (unsigned i = 0; i < config2->numQuads(); ++ i) {
        str << ' ' << config2->numAsicsStored(i);
      }
      for (unsigned iq = 0; iq != config2->numQuads(); ++ iq) {
        str << "\n  quad #" << iq;
        dumpQuadReg(str, config2->quads(iq));
      }
    }
    
  }

  shared_ptr<Psana::CsPad::ConfigV3> config3 = env.configStore().get(m_src);
  if (config3) {
    
    WithMsgLog(name(), info, str) {
      str << "CsPad::ConfigV3:";
      str << "\n  concentratorVersion = " << config3->concentratorVersion();
      str << "\n  runDelay = " << config3->runDelay();
      str << "\n  eventCode = " << config3->eventCode();
      str << "\n  protectionEnable = " << config3->protectionEnable();
      str << "\n  protectionThresholds:";
      for (unsigned i = 0; i < config3->numQuads(); ++ i) {
        const Psana::CsPad::ProtectionSystemThreshold& thr = config3->protectionThresholds()[i];
        str << "\n    adcThreshold=" << thr.adcThreshold()
            << " pixelCountThreshold=" << thr.pixelCountThreshold();
      }
      str << "\n  inactiveRunMode = " << config3->inactiveRunMode();
      str << "\n  activeRunMode = " << config3->activeRunMode();
      str << "\n  tdi = " << config3->tdi();
      str << "\n  payloadSize = " << config3->payloadSize();
      str << "\n  badAsicMask0 = " << config3->badAsicMask0();
      str << "\n  badAsicMask1 = " << config3->badAsicMask1();
      str << "\n  asicMask = " << config3->asicMask();
      str << "\n  quadMask = " << config3->quadMask();
      str << "\n  numAsicsRead = " << config3->numAsicsRead();
      str << "\n  numQuads = " << config3->numQuads();
      str << "\n  numSect = " << config3->numSect();
      str << "\n  roiMask =";
      for (unsigned i = 0; i < config3->numQuads(); ++ i) {
        str.setf(std::ios::showbase);
        str << ' ' << std::hex << config3->roiMask(i) << std::dec;
      }
      str << "\n  numAsicsStored =";
      for (unsigned i = 0; i < config3->numQuads(); ++ i) {
        str << ' ' << config3->numAsicsStored(i);
      }
      for (unsigned iq = 0; iq != config3->numQuads(); ++ iq) {
        str << "\n  quad #" << iq;
        dumpQuadReg(str, config3->quads(iq));
      }
    }
    
  }

  shared_ptr<Psana::CsPad::ConfigV4> config4 = env.configStore().get(m_src);
  if (config4) {
    
    WithMsgLog(name(), info, str) {
      str << "CsPad::ConfigV4:";
      str << "\n  concentratorVersion = " << config4->concentratorVersion();
      str << "\n  runDelay = " << config4->runDelay();
      str << "\n  eventCode = " << config4->eventCode();
      str << "\n  protectionEnable = " << config4->protectionEnable();
      str << "\n  protectionThresholds:";
      for (unsigned i = 0; i < config4->numQuads(); ++ i) {
        const Psana::CsPad::ProtectionSystemThreshold& thr = config4->protectionThresholds()[i];
        str << "\n    adcThreshold=" << thr.adcThreshold()
            << " pixelCountThreshold=" << thr.pixelCountThreshold();
      }
      str << "\n  inactiveRunMode = " << config4->inactiveRunMode();
      str << "\n  activeRunMode = " << config4->activeRunMode();
      str << "\n  tdi = " << config4->tdi();
      str << "\n  payloadSize = " << config4->payloadSize();
      str << "\n  badAsicMask0 = " << config4->badAsicMask0();
      str << "\n  badAsicMask1 = " << config4->badAsicMask1();
      str << "\n  asicMask = " << config4->asicMask();
      str << "\n  quadMask = " << config4->quadMask();
      str << "\n  numAsicsRead = " << config4->numAsicsRead();
      str << "\n  numQuads = " << config4->numQuads();
      str << "\n  numSect = " << config4->numSect();
      str << "\n  roiMask =";
      for (unsigned i = 0; i < config4->numQuads(); ++ i) {
        str.setf(std::ios::showbase);
        str << ' ' << std::hex << config4->roiMask(i) << std::dec;
      }
      str << "\n  numAsicsStored =";
      for (unsigned i = 0; i < config4->numQuads(); ++ i) {
        str << ' ' << config4->numAsicsStored(i);
      }
      for (unsigned iq = 0; iq != config4->numQuads(); ++ iq) {
        str << "\n  quad #" << iq;
        dumpQuadReg(str, config4->quads(iq));
      }
    }
    
  }

  shared_ptr<Psana::CsPad::ConfigV5> config5 = env.configStore().get(m_src);
  if (config5) {

    WithMsgLog(name(), info, str) {
      str << "CsPad::ConfigV5:";
      str << "\n  concentratorVersion = " << config5->concentratorVersion();
      str << "\n  runDelay = " << config5->runDelay();
      str << "\n  eventCode = " << config5->eventCode();
      str << "\n  protectionEnable = " << config5->protectionEnable();
      str << "\n  protectionThresholds:";
      for (unsigned i = 0; i < config5->numQuads(); ++ i) {
        const Psana::CsPad::ProtectionSystemThreshold& thr = config5->protectionThresholds()[i];
        str << "\n    adcThreshold=" << thr.adcThreshold()
            << " pixelCountThreshold=" << thr.pixelCountThreshold();
      }
      str << "\n  inactiveRunMode = " << config5->inactiveRunMode();
      str << "\n  activeRunMode = " << config5->activeRunMode();
      str << "\n  tdi = " << config5->tdi();
      str << "\n  payloadSize = " << config5->payloadSize();
      str << "\n  badAsicMask0 = " << config5->badAsicMask0();
      str << "\n  badAsicMask1 = " << config5->badAsicMask1();
      str << "\n  asicMask = " << config5->asicMask();
      str << "\n  quadMask = " << config5->quadMask();
      str << "\n  internalTriggerDelay = " << config5->internalTriggerDelay();
      str << "\n  numAsicsRead = " << config5->numAsicsRead();
      str << "\n  numQuads = " << config5->numQuads();
      str << "\n  numSect = " << config5->numSect();
      str << "\n  roiMask =";
      for (unsigned i = 0; i < config5->numQuads(); ++ i) {
        str.setf(std::ios::showbase);
        str << ' ' << std::hex << config5->roiMask(i) << std::dec;
      }
      str << "\n  numAsicsStored =";
      for (unsigned i = 0; i < config5->numQuads(); ++ i) {
        str << ' ' << config5->numAsicsStored(i);
      }
      for (unsigned iq = 0; iq != config5->numQuads(); ++ iq) {
        str << "\n  quad #" << iq;
        dumpQuadReg(str, config5->quads(iq));
      }
    }

  }
}

// Method which is called with event data
void 
DumpCsPad::event(Event& evt, Env& env)
{

  shared_ptr<Psana::CsPad::DataV1> data1 = evt.get(m_src, m_key);
  if (data1) {
    
    WithMsgLog(name(), info, str) {
      str << "CsPad::DataV1:";
      int nQuads = data1->quads_shape()[0];
      for (int q = 0; q < nQuads; ++ q) {
        const Psana::CsPad::ElementV1& el = data1->quads(q);
        str << "\n  Element #" << q ;
        str << "\n    virtual_channel = " << el.virtual_channel() ;
        str << "\n    lane = " << el.lane() ;
        str << "\n    tid = " << el.tid() ;
        str << "\n    acq_count = " << el.acq_count() ;
        str << "\n    op_code = " << el.op_code() ;
        str << "\n    quad = " << el.quad() ;
        str << "\n    seq_count = " << el.seq_count() ;
        str << "\n    ticks = " << el.ticks() ;
        str << "\n    fiducials = " << el.fiducials() ;
        str << "\n    frame_type = " << el.frame_type() ;
        str << "\n    sb_temp = " << el.sb_temp();

        const ndarray<const int16_t, 3>& data = el.data();
        str << "\n    common_mode = [ ";
        for (unsigned i = 0; i != data.shape()[0]; ++ i) {
            str << el.common_mode(i) << ' ';
        }
        str << "]";
        str << "\n    data = " << data;
      }

    }
  }


  shared_ptr<Psana::CsPad::DataV2> data2 = evt.get(m_src, m_key);
  if (data2) {
    
    WithMsgLog(name(), info, str) {
      str << "CsPad::DataV2:";
      int nQuads = data2->quads_shape()[0];
      for (int q = 0; q < nQuads; ++ q) {
        const Psana::CsPad::ElementV2& el = data2->quads(q);
        str << "\n  Element #" << q ;
        str << "\n    virtual_channel = " << el.virtual_channel() ;
        str << "\n    lane = " << el.lane() ;
        str << "\n    tid = " << el.tid() ;
        str << "\n    acq_count = " << el.acq_count() ;
        str << "\n    op_code = " << el.op_code() ;
        str << "\n    quad = " << el.quad() ;
        str << "\n    seq_count = " << el.seq_count() ;
        str << "\n    ticks = " << el.ticks() ;
        str << "\n    fiducials = " << el.fiducials() ;
        str << "\n    frame_type = " << el.frame_type() ;

        str << "\n    sb_temp = " << el.sb_temp();

        const ndarray<const int16_t, 3>& data = el.data();
        str << "\n    common_mode = [ ";
        for (unsigned i = 0; i != data.shape()[0]; ++ i) {
            str << el.common_mode(i) << ' ';
        }
        str << "]";
        str << "\n    data = " << data;
      }

    }
  }

}

} // namespace psana_examples
