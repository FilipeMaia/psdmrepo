//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpCsPad2x2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpCsPad2x2.h"

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
#include "psddl_psana/cspad2x2.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpCsPad2x2)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpCsPad2x2::DumpCsPad2x2 (const std::string& name)
  : Module(name)
{
  m_key = configStr("inputKey", "");
  m_src = configSrc("source", "DetInfo(:Cspad2x2)");
}

//--------------
// Destructor --
//--------------
DumpCsPad2x2::~DumpCsPad2x2 ()
{
}

// Method which is called at the beginning of the calibration cycle
void 
DumpCsPad2x2::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::CsPad2x2::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1) {
    
    WithMsgLog(name(), info, str) {
      str << "CsPad2x2::ConfigV1:";
      str << "\n  concentratorVersion = " << config1->concentratorVersion();
      str << "\n  protectionEnable = " << config1->protectionEnable();
      str << "\n  protectionThreshold:";
      str << "\n    adcThreshold= " << config1->protectionThreshold().adcThreshold()
          << "\n    pixelCountThreshold= " << config1->protectionThreshold().pixelCountThreshold();
      str << "\n  inactiveRunMode = " << config1->inactiveRunMode();
      str << "\n  activeRunMode = " << config1->activeRunMode();
      str << "\n  tdi = " << config1->tdi();
      str << "\n  payloadSize = " << config1->payloadSize();
      str << "\n  badAsicMask1 = " << config1->badAsicMask();
      str << "\n  asicMask = " << config1->asicMask();
      str << "\n  numAsicsRead = " << config1->numAsicsRead();
      str << "\n  roiMask = " << config1->roiMask();
      str << "\n  numAsicsStored = " << config1->numAsicsStored();
      const Psana::CsPad2x2::ConfigV1QuadReg& quad = config1->quad();
      str << "\n  quad:";
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
      str << "\n    PeltierEnable = " << quad.PeltierEnable();
      str << "\n    kpConstant = " << quad.kpConstant();
      str << "\n    kiConstant = " << quad.kiConstant();
      str << "\n    kdConstant = " << quad.kdConstant();
      str << "\n    humidThold = " << quad.humidThold();
      str << "\n    setPoint = " << quad.setPoint();
      str << "\n    digitalPots = " << quad.dp().pots();
      str << "\n    readOnly = shiftTest: " << quad.ro().shiftTest() << " verstion: " << quad.ro().version();
      str << "\n    gainMap = " << quad.gm().gainMap();
    }
    
  }

  shared_ptr<Psana::CsPad2x2::ConfigV2> config2 = env.configStore().get(m_src);
  if (config2) {

    WithMsgLog(name(), info, str) {
      str << "CsPad2x2::ConfigV2:";
      str << "\n  concentratorVersion = " << config2->concentratorVersion();
      str << "\n  protectionEnable = " << config2->protectionEnable();
      str << "\n  protectionThreshold:";
      str << "\n    adcThreshold= " << config2->protectionThreshold().adcThreshold()
          << "\n    pixelCountThreshold= " << config2->protectionThreshold().pixelCountThreshold();
      str << "\n  inactiveRunMode = " << config2->inactiveRunMode();
      str << "\n  activeRunMode = " << config2->activeRunMode();
      str << "\n  runTriggerDelay = " << config2->runTriggerDelay();
      str << "\n  tdi = " << config2->tdi();
      str << "\n  payloadSize = " << config2->payloadSize();
      str << "\n  badAsicMask1 = " << config2->badAsicMask();
      str << "\n  asicMask = " << config2->asicMask();
      str << "\n  numAsicsRead = " << config2->numAsicsRead();
      str << "\n  roiMask = " << config2->roiMask();
      str << "\n  numAsicsStored = " << config2->numAsicsStored();
      const Psana::CsPad2x2::ConfigV2QuadReg& quad = config2->quad();
      str << "\n  quad:";
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
      str << "\n    PeltierEnable = " << quad.PeltierEnable();
      str << "\n    kpConstant = " << quad.kpConstant();
      str << "\n    kiConstant = " << quad.kiConstant();
      str << "\n    kdConstant = " << quad.kdConstant();
      str << "\n    humidThold = " << quad.humidThold();
      str << "\n    setPoint = " << quad.setPoint();
      str << "\n    digitalPots = " << quad.dp().pots();
      str << "\n    readOnly = shiftTest: " << quad.ro().shiftTest() << " verstion: " << quad.ro().version();
      str << "\n    gainMap = " << quad.gm().gainMap();
    }

  }
}

// Method which is called with event data
void 
DumpCsPad2x2::event(Event& evt, Env& env)
{
  shared_ptr<Psana::CsPad2x2::ElementV1> elem1 = evt.get(m_src, m_key);
  if (elem1) {

    WithMsgLog(name(), info, str) {
      str << "CsPad2x2::ElementV1:";
      str << "\n  virtual_channel = " << elem1->virtual_channel() ;
      str << "\n  lane = " << elem1->lane() ;
      str << "\n  tid = " << elem1->tid() ;
      str << "\n  acq_count = " << elem1->acq_count() ;
      str << "\n  op_code = " << elem1->op_code() ;
      str << "\n  quad = " << elem1->quad() ;
      str << "\n  seq_count = " << elem1->seq_count() ;
      str << "\n  ticks = " << elem1->ticks() ;
      str << "\n  fiducials = " << elem1->fiducials() ;
      str << "\n  frame_type = " << elem1->frame_type() ;

      str << "\n    sb_temp = " << elem1->sb_temp();

      const ndarray<const int16_t, 3>& data = elem1->data();
      str << "\n    common_mode = [ ";
      for (unsigned i = 0; i != data.shape()[2]; ++ i) {
          str << elem1->common_mode(i) << ' ';
      }
      str << "]";
      str << "\n    data = " << data;
    }
  }

}

} // namespace psana_examples
