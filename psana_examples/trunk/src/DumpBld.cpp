//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpBld...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpBld.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/bld.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpBld)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpBld::DumpBld (const std::string& name)
  : Module(name)
{
  m_ebeamSrc = configStr("eBeamSource", "BldInfo(EBeam)");
  m_cavSrc = configStr("phaseCavSource", "BldInfo(PhaseCavity)");
  m_feeSrc = configStr("feeSource", "BldInfo(FEEGasDetEnergy)");
  m_ipimbSrc = configStr("feeSource", "BldInfo(NH2-SB1-IPM-01)");
}

//--------------
// Destructor --
//--------------
DumpBld::~DumpBld ()
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpBld::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Bld::BldDataEBeamV0> ebeam0 = evt.get(m_ebeamSrc);
  if (ebeam0.get()) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataEBeamV0:"
          << "\n  damageMask=" << ebeam0->damageMask()
          << "\n  ebeamCharge=" << ebeam0->ebeamCharge()
          << "\n  ebeamL3Energy=" << ebeam0->ebeamL3Energy()
          << "\n  ebeamLTUPosX=" << ebeam0->ebeamLTUPosX()
          << "\n  ebeamLTUPosY=" << ebeam0->ebeamLTUPosY()
          << "\n  ebeamLTUAngX=" << ebeam0->ebeamLTUAngX()
          << "\n  ebeamLTUAngY=" << ebeam0->ebeamLTUAngY();
    }
  }

  shared_ptr<Psana::Bld::BldDataEBeam> ebeam = evt.get(m_ebeamSrc);
  if (ebeam.get()) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataEBeam:"
          << "\n  damageMask=" << ebeam->damageMask()
          << "\n  ebeamCharge=" << ebeam->ebeamCharge()
          << "\n  ebeamL3Energy=" << ebeam->ebeamL3Energy()
          << "\n  ebeamLTUPosX=" << ebeam->ebeamLTUPosX()
          << "\n  ebeamLTUPosY=" << ebeam->ebeamLTUPosY()
          << "\n  ebeamLTUAngX=" << ebeam->ebeamLTUAngX()
          << "\n  ebeamLTUAngY=" << ebeam->ebeamLTUAngY()
          << "\n  ebeamPkCurrBC2=" << ebeam->ebeamPkCurrBC2();
    }
  }

  shared_ptr<Psana::Bld::BldDataPhaseCavity> cav = evt.get(m_cavSrc);
  if (cav.get()) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataPhaseCavity:" 
          << "\n  fitTime1=" << cav->fitTime1()
          << "\n  fitTime2=" << cav->fitTime2()
          << "\n  charge1=" << cav->charge1()
          << "\n  charge2=" << cav->charge2();
    }
  }
  
  shared_ptr<Psana::Bld::BldDataFEEGasDetEnergy> fee = evt.get(m_feeSrc);
  if (fee.get()) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataFEEGasDetEnergy:"
          << "\n  f_11_ENRC=" << fee->f_11_ENRC()
          << "\n  f_12_ENRC=" << fee->f_12_ENRC()
          << "\n  f_21_ENRC=" << fee->f_21_ENRC()
          << "\n  f_22_ENRC=" << fee->f_22_ENRC();
    }
  }

  shared_ptr<Psana::Bld::BldDataIpimb> ipimb = evt.get(m_ipimbSrc);
  if (ipimb.get()) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataIpimb:";
      const Psana::Ipimb::DataV1& ipimbData = ipimb->ipimbData();
      str << "\n  Ipimb::DataV1:"
          << "\n    triggerCounter = " << ipimbData.triggerCounter()
          << "\n    config = " << ipimbData.config0()
          << "," << ipimbData.config1()
          << "," << ipimbData.config2()
          << "\n    channel = " << ipimbData.channel0()
          << "," << ipimbData.channel1()
          << "," << ipimbData.channel2()
          << "," << ipimbData.channel3()
          << "\n    volts = " << ipimbData.channel0Volts()
          << "," << ipimbData.channel1Volts()
          << "," << ipimbData.channel2Volts()
          << "," << ipimbData.channel3Volts()
          << "\n    checksum = " << ipimbData.checksum();
      
      const Psana::Ipimb::ConfigV1& ipimbConfig = ipimb->ipimbConfig();
      str << "\n  Ipimb::ConfigV1:";
      str << "\n    triggerCounter = " << ipimbConfig.triggerCounter();
      str << "\n    serialID = " << ipimbConfig.serialID();
      str << "\n    chargeAmpRange = " << ipimbConfig.chargeAmpRange();
      str << "\n    calibrationRange = " << ipimbConfig.calibrationRange();
      str << "\n    resetLength = " << ipimbConfig.resetLength();
      str << "\n    resetDelay = " << ipimbConfig.resetDelay();
      str << "\n    chargeAmpRefVoltage = " << ipimbConfig.chargeAmpRefVoltage();
      str << "\n    calibrationVoltage = " << ipimbConfig.calibrationVoltage();
      str << "\n    diodeBias = " << ipimbConfig.diodeBias();
      str << "\n    status = " << ipimbConfig.status();
      str << "\n    errors = " << ipimbConfig.errors();
      str << "\n    calStrobeLength = " << ipimbConfig.calStrobeLength();
      str << "\n    trigDelay = " << ipimbConfig.trigDelay();
      
      const Psana::Lusi::IpmFexV1& ipmFexData = ipimb->ipmFexData();
      str << "\n  Psana::Lusi::IpmFexV1:";
      str << "\n    sum = " << ipmFexData.sum();
      str << "\n    xpos = " << ipmFexData.xpos();
      str << "\n    ypos = " << ipmFexData.ypos();
      const float* channel = ipmFexData.channel();
      str << "\n    channel =";
      for (int i = 0; i < Psana::Lusi::IpmFexV1::NCHANNELS; ++ i) {
        str << " " << channel[i];
      }
    }
  }
}

} // namespace psana_examples
