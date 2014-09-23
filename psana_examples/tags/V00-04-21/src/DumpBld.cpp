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
#include <iomanip>

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
  m_ebeamSrc = configSrc("eBeamSource", "BldInfo(EBeam)");
  m_cavSrc = configSrc("phaseCavSource", "BldInfo(PhaseCavity)");
  m_feeSrc = configSrc("feeSource", "BldInfo(FEEGasDetEnergy)");
  m_ipimbSrc = configSrc("ipimbSource", "BldInfo(NH2-SB1-IPM-01)");
  m_pimSrc = configSrc("pimSource", "BldInfo(XCS-DIO-01)");
  m_gmdSrc = configSrc("gmdSource", "BldInfo()");
}

//--------------
// Destructor --
//--------------
DumpBld::~DumpBld ()
{
}

void
DumpBld::beginJob(Event& evt, Env& env)
{
  event(evt, env);
}

// Method which is called with event data
void 
DumpBld::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Bld::BldDataEBeamV0> ebeam0 = evt.get(m_ebeamSrc);
  if (ebeam0) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataEBeamV0:"
          << "\n  damageMask=" << std::showbase << std::hex << ebeam0->damageMask() << std::dec
          << "\n  ebeamCharge=" << ebeam0->ebeamCharge()
          << "\n  ebeamL3Energy=" << ebeam0->ebeamL3Energy()
          << "\n  ebeamLTUPosX=" << ebeam0->ebeamLTUPosX()
          << "\n  ebeamLTUPosY=" << ebeam0->ebeamLTUPosY()
          << "\n  ebeamLTUAngX=" << ebeam0->ebeamLTUAngX()
          << "\n  ebeamLTUAngY=" << ebeam0->ebeamLTUAngY();
    }
  }

  shared_ptr<Psana::Bld::BldDataEBeamV1> ebeam1 = evt.get(m_ebeamSrc);
  if (ebeam1) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataEBeamV1:"
          << "\n  damageMask=" << std::showbase << std::hex << ebeam1->damageMask() << std::dec
          << "\n  ebeamCharge=" << ebeam1->ebeamCharge()
          << "\n  ebeamL3Energy=" << ebeam1->ebeamL3Energy()
          << "\n  ebeamLTUPosX=" << ebeam1->ebeamLTUPosX()
          << "\n  ebeamLTUPosY=" << ebeam1->ebeamLTUPosY()
          << "\n  ebeamLTUAngX=" << ebeam1->ebeamLTUAngX()
          << "\n  ebeamLTUAngY=" << ebeam1->ebeamLTUAngY()
          << "\n  ebeamPkCurrBC2=" << ebeam1->ebeamPkCurrBC2();
    }
  }

  shared_ptr<Psana::Bld::BldDataEBeamV2> ebeam2 = evt.get(m_ebeamSrc);
  if (ebeam2) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataEBeamV2:"
          << "\n  damageMask=" << std::showbase << std::hex << ebeam2->damageMask() << std::dec
          << "\n  ebeamCharge=" << ebeam2->ebeamCharge()
          << "\n  ebeamL3Energy=" << ebeam2->ebeamL3Energy()
          << "\n  ebeamLTUPosX=" << ebeam2->ebeamLTUPosX()
          << "\n  ebeamLTUPosY=" << ebeam2->ebeamLTUPosY()
          << "\n  ebeamLTUAngX=" << ebeam2->ebeamLTUAngX()
          << "\n  ebeamLTUAngY=" << ebeam2->ebeamLTUAngY()
          << "\n  ebeamPkCurrBC2=" << ebeam2->ebeamPkCurrBC2()
          << "\n  ebeamEnergyBC2=" << ebeam2->ebeamEnergyBC2();
    }
  }

  shared_ptr<Psana::Bld::BldDataEBeamV3> ebeam3 = evt.get(m_ebeamSrc);
  if (ebeam3) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataEBeamV3:"
          << "\n  damageMask=" << std::showbase << std::hex << ebeam3->damageMask() << std::dec
          << "\n  ebeamCharge=" << ebeam3->ebeamCharge()
          << "\n  ebeamL3Energy=" << ebeam3->ebeamL3Energy()
          << "\n  ebeamLTUPosX=" << ebeam3->ebeamLTUPosX()
          << "\n  ebeamLTUPosY=" << ebeam3->ebeamLTUPosY()
          << "\n  ebeamLTUAngX=" << ebeam3->ebeamLTUAngX()
          << "\n  ebeamLTUAngY=" << ebeam3->ebeamLTUAngY()
          << "\n  ebeamPkCurrBC2=" << ebeam3->ebeamPkCurrBC2()
          << "\n  ebeamEnergyBC2=" << ebeam3->ebeamEnergyBC2()
          << "\n  ebeamPkCurrBC1=" << ebeam3->ebeamPkCurrBC1()
          << "\n  ebeamEnergyBC1=" << ebeam3->ebeamEnergyBC1();
    }
  }

  shared_ptr<Psana::Bld::BldDataEBeamV4> ebeam4 = evt.get(m_ebeamSrc);
  if (ebeam4) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataEBeamV4:"
          << "\n  damageMask=" << std::showbase << std::hex << ebeam4->damageMask() << std::dec
          << "\n  ebeamCharge=" << ebeam4->ebeamCharge()
          << "\n  ebeamL3Energy=" << ebeam4->ebeamL3Energy()
          << "\n  ebeamLTUPosX=" << ebeam4->ebeamLTUPosX()
          << "\n  ebeamLTUPosY=" << ebeam4->ebeamLTUPosY()
          << "\n  ebeamLTUAngX=" << ebeam4->ebeamLTUAngX()
          << "\n  ebeamLTUAngY=" << ebeam4->ebeamLTUAngY()
          << "\n  ebeamPkCurrBC2=" << ebeam4->ebeamPkCurrBC2()
          << "\n  ebeamEnergyBC2=" << ebeam4->ebeamEnergyBC2()
          << "\n  ebeamPkCurrBC1=" << ebeam4->ebeamPkCurrBC1()
          << "\n  ebeamEnergyBC1=" << ebeam4->ebeamEnergyBC1()
          << "\n  ebeamUndPosX=" << ebeam4->ebeamUndPosX()
          << "\n  ebeamUndPosY=" << ebeam4->ebeamUndPosY()
          << "\n  ebeamUndAngX=" << ebeam4->ebeamUndAngX()
          << "\n  ebeamUndAngY=" << ebeam4->ebeamUndAngY();
    }
  }

  shared_ptr<Psana::Bld::BldDataEBeamV5> ebeam5 = evt.get(m_ebeamSrc);
  if (ebeam5) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataEBeamV5:"
          << "\n  damageMask=" << std::showbase << std::hex << ebeam5->damageMask() << std::dec
          << "\n  ebeamCharge=" << ebeam5->ebeamCharge()
          << "\n  ebeamL3Energy=" << ebeam5->ebeamL3Energy()
          << "\n  ebeamLTUPosX=" << ebeam5->ebeamLTUPosX()
          << "\n  ebeamLTUPosY=" << ebeam5->ebeamLTUPosY()
          << "\n  ebeamLTUAngX=" << ebeam5->ebeamLTUAngX()
          << "\n  ebeamLTUAngY=" << ebeam5->ebeamLTUAngY()
          << "\n  ebeamPkCurrBC2=" << ebeam5->ebeamPkCurrBC2()
          << "\n  ebeamEnergyBC2=" << ebeam5->ebeamEnergyBC2()
          << "\n  ebeamPkCurrBC1=" << ebeam5->ebeamPkCurrBC1()
          << "\n  ebeamEnergyBC1=" << ebeam5->ebeamEnergyBC1()
          << "\n  ebeamUndPosX=" << ebeam5->ebeamUndPosX()
          << "\n  ebeamUndPosY=" << ebeam5->ebeamUndPosY()
          << "\n  ebeamUndAngX=" << ebeam5->ebeamUndAngX()
          << "\n  ebeamUndAngY=" << ebeam5->ebeamUndAngY()
          << "\n  ebeamXTCAVAmpl=" << ebeam5->ebeamXTCAVAmpl()
          << "\n  ebeamXTCAVPhase=" << ebeam5->ebeamXTCAVPhase()
          << "\n  ebeamDumpCharge=" << ebeam5->ebeamDumpCharge();
    }
  }

  shared_ptr<Psana::Bld::BldDataPhaseCavity> cav = evt.get(m_cavSrc);
  if (cav) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataPhaseCavity:" 
          << "\n  fitTime1=" << cav->fitTime1()
          << "\n  fitTime2=" << cav->fitTime2()
          << "\n  charge1=" << cav->charge1()
          << "\n  charge2=" << cav->charge2();
    }
  }
  
  shared_ptr<Psana::Bld::BldDataFEEGasDetEnergyV1> feeV1 = evt.get(m_feeSrc);
  if (feeV1) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataFEEGasDetEnergyV1:"
          << "\n  f_11_ENRC=" << feeV1->f_11_ENRC()
          << "\n  f_12_ENRC=" << feeV1->f_12_ENRC()
          << "\n  f_21_ENRC=" << feeV1->f_21_ENRC()
          << "\n  f_22_ENRC=" << feeV1->f_22_ENRC()
          << "\n  f_63_ENRC=" << feeV1->f_63_ENRC()
          << "\n  f_64_ENRC=" << feeV1->f_64_ENRC();
    }
  } else {
    shared_ptr<Psana::Bld::BldDataFEEGasDetEnergy> fee = evt.get(m_feeSrc);
    if (fee) {
      WithMsgLog(name(), info, str) {
        str << "Bld::BldDataFEEGasDetEnergy:"
            << "\n  f_11_ENRC=" << fee->f_11_ENRC()
            << "\n  f_12_ENRC=" << fee->f_12_ENRC()
            << "\n  f_21_ENRC=" << fee->f_21_ENRC()
            << "\n  f_22_ENRC=" << fee->f_22_ENRC();
      }
    }
  }

  shared_ptr<Psana::Bld::BldDataIpimbV0> ipimb0 = evt.get(m_ipimbSrc);
  if (ipimb0) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataIpimbV0:";
      const Psana::Ipimb::DataV1& ipimbData = ipimb0->ipimbData();
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
      
      const Psana::Ipimb::ConfigV1& ipimbConfig = ipimb0->ipimbConfig();
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
      
      const Psana::Lusi::IpmFexV1& ipmFexData = ipimb0->ipmFexData();
      str << "\n  Psana::Lusi::IpmFexV1:";
      str << "\n    sum = " << ipmFexData.sum();
      str << "\n    xpos = " << ipmFexData.xpos();
      str << "\n    ypos = " << ipmFexData.ypos();
      str << "\n    channel = " << ipmFexData.channel();
    }
  }

  shared_ptr<Psana::Bld::BldDataIpimbV1> ipimb1 = evt.get(m_ipimbSrc);
  if (ipimb1) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataIpimbV1:";
      const Psana::Ipimb::DataV2& ipimbData = ipimb1->ipimbData();
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
          << "\n    channel-ps = " << ipimbData.channel0ps()
          << "," << ipimbData.channel1ps()
          << "," << ipimbData.channel2ps()
          << "," << ipimbData.channel3ps()
          << "\n    volts-ps = " << ipimbData.channel0psVolts()
          << "," << ipimbData.channel1psVolts()
          << "," << ipimbData.channel2psVolts()
          << "," << ipimbData.channel3psVolts()
          << "\n    checksum = " << ipimbData.checksum();
      
      const Psana::Ipimb::ConfigV2& ipimbConfig = ipimb1->ipimbConfig();
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
      str << "\n    trigPsDelay = " << ipimbConfig.trigPsDelay();
      str << "\n    adcDelay = " << ipimbConfig.adcDelay();
      
      const Psana::Lusi::IpmFexV1& ipmFexData = ipimb1->ipmFexData();
      str << "\n  Psana::Lusi::IpmFexV1:";
      str << "\n    sum = " << ipmFexData.sum();
      str << "\n    xpos = " << ipmFexData.xpos();
      str << "\n    ypos = " << ipmFexData.ypos();
      str << "\n    channel = " << ipmFexData.channel();
    }
  }

  shared_ptr<Psana::Bld::BldDataPimV1> pim1 = evt.get(m_pimSrc);
  if (pim1) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataPimV1:";
      const Psana::Pulnix::TM6740ConfigV2& camCfg = pim1->camConfig();
      str << "\n  Pulnix::TM6740ConfigV2:"
          << "\n    vref_a = " << camCfg.vref_a()
          << "\n    vref_b = " << camCfg.vref_b()
          << "\n    gain_a = " << camCfg.gain_a()
          << "\n    gain_b = " << camCfg.gain_b()
          << "\n    gain_balance = " << (camCfg.gain_balance() ? "yes" : "no")
          << "\n    output_resolution = " << camCfg.output_resolution()
          << "\n    output_resolution_bits = " << camCfg.output_resolution_bits()
          << "\n    horizontal_binning = " << camCfg.horizontal_binning()
          << "\n    vertical_binning = " << camCfg.vertical_binning()
          << "\n    lookuptable_mode = " << camCfg.lookuptable_mode();

      const Psana::Lusi::PimImageConfigV1& pimCfg = pim1->pimConfig();
      str << "\n  Lusi::PimImageConfigV1:"
          << "\n    xscale = " << pimCfg.xscale()
          << "\n    yscale = " << pimCfg.yscale();

      const Psana::Camera::FrameV1& frame = pim1->frame();
      str << "\n  Camera::FrameV1:"
          << "\n    width=" << frame.width()
          << "\n    height=" << frame.height()
          << "\n    depth=" << frame.depth()
          << "\n    offset=" << frame.offset();

      const ndarray<const uint8_t, 2>& data8 = frame.data8();
      if (not data8.empty()) {
        str << "\n    data8=" << data8;
      }

      const ndarray<const uint16_t, 2>& data16 = frame.data16();
      if (not data16.empty()) {
        str << "\n    data16=" << data16;
      }
    }
  }


  // dump BldDataGMDV0
  shared_ptr<Psana::Bld::BldDataGMDV0> gmd0 = evt.get(m_gmdSrc);
  if (gmd0) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataGMDV0:"
          << "\n  gasType = " << gmd0->gasType()
          << "\n  pressure = " << gmd0->pressure()
          << "\n  temperature = " << gmd0->temperature()
          << "\n  current = " << gmd0->current()
          << "\n  hvMeshElectron = " << gmd0->hvMeshElectron()
          << "\n  hvMeshIon = " << gmd0->hvMeshIon()
          << "\n  hvMultIon = " << gmd0->hvMultIon()
          << "\n  chargeQ = " << gmd0->chargeQ()
          << "\n  photonEnergy = " << gmd0->photonEnergy()
          << "\n  multPulseIntensity = " << gmd0->multPulseIntensity()
          << "\n  keithleyPulseIntensity = " << gmd0->keithleyPulseIntensity()
          << "\n  pulseEnergy = " << gmd0->pulseEnergy()
          << "\n  pulseEnergyFEE = " << gmd0->pulseEnergyFEE()
          << "\n  transmission = " << gmd0->transmission()
          << "\n  transmissionFEE = " << gmd0->transmissionFEE();
    }
  }

  // dump BldDataGMDV1
  shared_ptr<Psana::Bld::BldDataGMDV1> gmd1 = evt.get(m_gmdSrc);
  if (gmd1) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataGMDV1:"
          << "\n  milliJoulesPerPulse = " << gmd1->milliJoulesPerPulse()
          << "\n  milliJoulesAverage = " << gmd1->milliJoulesAverage()
          << "\n  correctedSumPerPulse = " << gmd1->correctedSumPerPulse()
          << "\n  bgValuePerSample = " << gmd1->bgValuePerSample()
          << "\n  relativeEnergyPerPulse = " << gmd1->relativeEnergyPerPulse();
    }
  }

  // dump BldDataSpectrometerV0
  shared_ptr<Psana::Bld::BldDataSpectrometerV0> spec0 = evt.get(m_gmdSrc);
  if (spec0) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataSpectrometerV0:"
          << "\n  hproj = " << spec0->hproj()
          << "\n  vproj = " << spec0->vproj();
    }
  }

}

} // namespace psana_examples
