
/* Do not modify this file, it is generated code. 
   Modify the template file and Ddl backend and run the psddl compiler. */

#include "psddl_psana/acqiris.ddl.h"
#include "psddl_psana/alias.ddl.h"
#include "psddl_psana/andor.ddl.h"
#include "psddl_psana/bld.ddl.h"
#include "psddl_psana/camera.ddl.h"
#include "psddl_psana/control.ddl.h"
#include "psddl_psana/cspad.ddl.h"
#include "psddl_psana/cspad2x2.ddl.h"
#include "psddl_psana/encoder.ddl.h"
#include "psddl_psana/epix.ddl.h"
#include "psddl_psana/epixsampler.ddl.h"
#include "psddl_psana/evr.ddl.h"
#include "psddl_psana/fccd.ddl.h"
#include "psddl_psana/fli.ddl.h"
#include "psddl_psana/gsc16ai.ddl.h"
#include "psddl_psana/imp.ddl.h"
#include "psddl_psana/ipimb.ddl.h"
#include "psddl_psana/l3t.ddl.h"
#include "psddl_psana/lusi.ddl.h"
#include "psddl_psana/oceanoptics.ddl.h"
#include "psddl_psana/opal1k.ddl.h"
#include "psddl_psana/orca.ddl.h"
#include "psddl_psana/pnccd.ddl.h"
#include "psddl_psana/princeton.ddl.h"
#include "psddl_psana/pulnix.ddl.h"
#include "psddl_psana/quartz.ddl.h"
#include "psddl_psana/rayonix.ddl.h"
#include "psddl_psana/timepix.ddl.h"
#include "psddl_psana/usdusb.ddl.h"

#include "MsgLogger/MsgLogger.h"
#include "PSEvt/Event.h"
#include "PSEnv/Env.h"
#include "PsanaTools/psddl_Dump.h"
#include <cstdio>
#include <vector>

using namespace std;
using namespace Psana;

namespace {
const char * logger = "psddl_Dump";
}

namespace PrintPsddl {

// helper function to indent lines. 
// Uses global variable that print functions set.
int INDENT = 0;
void indent() {
  for (int idx=0; idx < INDENT; ++idx) printf(" ");
}

// helper function to get a small set of indicies to print
// for big arrays.
vector<int> arrayInds(size_t arraySize, int numInds = 10) {
  if (arraySize <= size_t(numInds)) {
    vector<int> inds(arraySize);
    for (unsigned idx=0; idx < arraySize; ++idx) inds[idx]=idx;
    return inds;
  }
  vector<int> inds(numInds);
  for (int idx=0; idx < numInds; ++idx) {
    inds[idx]=idx*arraySize;
  }
  return inds;
}

// declare print functions for all xtc types and their subtypes
void print(const Acqiris::ConfigV1 &obj);
void print(const Acqiris::DataDescV1 &obj);
void print(const Acqiris::DataDescV1Elem &obj);
void print(const Acqiris::HorizV1 &obj);
void print(const Acqiris::TdcAuxIO &obj);
void print(const Acqiris::TdcChannel &obj);
void print(const Acqiris::TdcConfigV1 &obj);
void print(const Acqiris::TdcDataV1 &obj);
void print(const Acqiris::TdcDataV1Channel &obj);
void print(const Acqiris::TdcDataV1Common &obj);
void print(const Acqiris::TdcDataV1Marker &obj);
void print(const Acqiris::TdcDataV1_Item &obj);
void print(const Acqiris::TdcVetoIO &obj);
void print(const Acqiris::TimestampV1 &obj);
void print(const Acqiris::TrigV1 &obj);
void print(const Acqiris::VertV1 &obj);
void print(const Alias::ConfigV1 &obj);
void print(const Alias::SrcAlias &obj);
void print(const Andor::ConfigV1 &obj);
void print(const Andor::FrameV1 &obj);
void print(const Bld::BldDataAcqADCV1 &obj);
void print(const Bld::BldDataEBeamV0 &obj);
void print(const Bld::BldDataEBeamV1 &obj);
void print(const Bld::BldDataEBeamV2 &obj);
void print(const Bld::BldDataEBeamV3 &obj);
void print(const Bld::BldDataEBeamV4 &obj);
void print(const Bld::BldDataFEEGasDetEnergy &obj);
void print(const Bld::BldDataGMDV0 &obj);
void print(const Bld::BldDataGMDV1 &obj);
void print(const Bld::BldDataIpimbV0 &obj);
void print(const Bld::BldDataIpimbV1 &obj);
void print(const Bld::BldDataPhaseCavity &obj);
void print(const Bld::BldDataPimV1 &obj);
void print(const Bld::BldDataSpectrometerV0 &obj);
void print(const Camera::FrameCoord &obj);
void print(const Camera::FrameFccdConfigV1 &obj);
void print(const Camera::FrameFexConfigV1 &obj);
void print(const Camera::FrameV1 &obj);
void print(const Camera::TwoDGaussianV1 &obj);
void print(const ControlData::ConfigV1 &obj);
void print(const ControlData::ConfigV2 &obj);
void print(const ControlData::ConfigV3 &obj);
void print(const ControlData::PVControl &obj);
void print(const ControlData::PVLabel &obj);
void print(const ControlData::PVMonitor &obj);
void print(const CsPad::ConfigV1 &obj);
void print(const CsPad::ConfigV1QuadReg &obj);
void print(const CsPad::ConfigV2 &obj);
void print(const CsPad::ConfigV2QuadReg &obj);
void print(const CsPad::ConfigV3 &obj);
void print(const CsPad::ConfigV3QuadReg &obj);
void print(const CsPad::ConfigV4 &obj);
void print(const CsPad::ConfigV5 &obj);
void print(const CsPad::CsPadDigitalPotsCfg &obj);
void print(const CsPad::CsPadGainMapCfg &obj);
void print(const CsPad::CsPadReadOnlyCfg &obj);
void print(const CsPad::DataV1 &obj);
void print(const CsPad::DataV2 &obj);
void print(const CsPad::ElementV1 &obj);
void print(const CsPad::ElementV2 &obj);
void print(const CsPad::ProtectionSystemThreshold &obj);
void print(const CsPad2x2::ConfigV1 &obj);
void print(const CsPad2x2::ConfigV1QuadReg &obj);
void print(const CsPad2x2::ConfigV2 &obj);
void print(const CsPad2x2::ConfigV2QuadReg &obj);
void print(const CsPad2x2::CsPad2x2DigitalPotsCfg &obj);
void print(const CsPad2x2::CsPad2x2GainMapCfg &obj);
void print(const CsPad2x2::CsPad2x2ReadOnlyCfg &obj);
void print(const CsPad2x2::ElementV1 &obj);
void print(const CsPad2x2::ProtectionSystemThreshold &obj);
void print(const Encoder::ConfigV1 &obj);
void print(const Encoder::ConfigV2 &obj);
void print(const Encoder::DataV1 &obj);
void print(const Encoder::DataV2 &obj);
void print(const Epix::AsicConfigV1 &obj);
void print(const Epix::ConfigV1 &obj);
void print(const Epix::ElementV1 &obj);
void print(const EpixSampler::ConfigV1 &obj);
void print(const EpixSampler::ElementV1 &obj);
void print(const EvrData::ConfigV1 &obj);
void print(const EvrData::ConfigV2 &obj);
void print(const EvrData::ConfigV3 &obj);
void print(const EvrData::ConfigV4 &obj);
void print(const EvrData::ConfigV5 &obj);
void print(const EvrData::ConfigV6 &obj);
void print(const EvrData::ConfigV7 &obj);
void print(const EvrData::DataV3 &obj);
void print(const EvrData::EventCodeV3 &obj);
void print(const EvrData::EventCodeV4 &obj);
void print(const EvrData::EventCodeV5 &obj);
void print(const EvrData::EventCodeV6 &obj);
void print(const EvrData::FIFOEvent &obj);
void print(const EvrData::IOChannel &obj);
void print(const EvrData::IOConfigV1 &obj);
void print(const EvrData::OutputMap &obj);
void print(const EvrData::OutputMapV2 &obj);
void print(const EvrData::PulseConfig &obj);
void print(const EvrData::PulseConfigV3 &obj);
void print(const EvrData::SequencerConfigV1 &obj);
void print(const EvrData::SequencerEntry &obj);
void print(const EvrData::SrcConfigV1 &obj);
void print(const EvrData::SrcEventCode &obj);
void print(const FCCD::FccdConfigV1 &obj);
void print(const FCCD::FccdConfigV2 &obj);
void print(const Fli::ConfigV1 &obj);
void print(const Fli::FrameV1 &obj);
void print(const Gsc16ai::ConfigV1 &obj);
void print(const Gsc16ai::DataV1 &obj);
void print(const Imp::ConfigV1 &obj);
void print(const Imp::ElementV1 &obj);
void print(const Imp::LaneStatus &obj);
void print(const Imp::Sample &obj);
void print(const Ipimb::ConfigV1 &obj);
void print(const Ipimb::ConfigV2 &obj);
void print(const Ipimb::DataV1 &obj);
void print(const Ipimb::DataV2 &obj);
void print(const L3T::ConfigV1 &obj);
void print(const L3T::DataV1 &obj);
void print(const Lusi::DiodeFexConfigV1 &obj);
void print(const Lusi::DiodeFexConfigV2 &obj);
void print(const Lusi::DiodeFexV1 &obj);
void print(const Lusi::IpmFexConfigV1 &obj);
void print(const Lusi::IpmFexConfigV2 &obj);
void print(const Lusi::IpmFexV1 &obj);
void print(const Lusi::PimImageConfigV1 &obj);
void print(const OceanOptics::ConfigV1 &obj);
void print(const OceanOptics::DataV1 &obj);
void print(const OceanOptics::timespec64 &obj);
void print(const Opal1k::ConfigV1 &obj);
void print(const Orca::ConfigV1 &obj);
void print(const PNCCD::ConfigV1 &obj);
void print(const PNCCD::ConfigV2 &obj);
void print(const PNCCD::FrameV1 &obj);
void print(const PNCCD::FramesV1 &obj);
void print(const PNCCD::FullFrameV1 &obj);
void print(const Pds::ClockTime &obj);
void print(const Pds::DetInfo &obj);
void print(const Pds::Src &obj);
void print(const Princeton::ConfigV1 &obj);
void print(const Princeton::ConfigV2 &obj);
void print(const Princeton::ConfigV3 &obj);
void print(const Princeton::ConfigV4 &obj);
void print(const Princeton::ConfigV5 &obj);
void print(const Princeton::FrameV1 &obj);
void print(const Princeton::FrameV2 &obj);
void print(const Princeton::InfoV1 &obj);
void print(const Pulnix::TM6740ConfigV1 &obj);
void print(const Pulnix::TM6740ConfigV2 &obj);
void print(const Quartz::ConfigV1 &obj);
void print(const Rayonix::ConfigV1 &obj);
void print(const Rayonix::ConfigV2 &obj);
void print(const Timepix::ConfigV1 &obj);
void print(const Timepix::ConfigV2 &obj);
void print(const Timepix::ConfigV3 &obj);
void print(const Timepix::DataV1 &obj);
void print(const Timepix::DataV2 &obj);
void print(const UsdUsb::ConfigV1 &obj);
void print(const UsdUsb::DataV1 &obj);

// declare print functions for the basic types
void print(const int8_t d) {
  printf("0x%2.2X",d);
}

void print(const int16_t d) {
  printf("0x%4.4X",d);
}

void print(const int32_t d) {
  printf("0x%8.8X",d);
}

void print(const int64_t d) {
  printf("0x%16.16lX",d);
}

void print(const uint8_t d) {
  printf("0x%2.2X",d);
}

void print(const uint16_t d) {
  printf("0x%4.4X",d);
}

void print(const uint32_t d) {
  printf("0x%8.8X",d);
}

void print(const uint64_t d) {
  printf("0x%16.16lX",d);
}

void print(const double d) {
  printf("%.5e",d);
}

void print(const float d) {
  printf("%.5e",d);
}


// define print functions for all xtc types and subtypes.
void print(const Acqiris::ConfigV1 &obj) {
  printf("Acqiris ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("nbrConvertersPerChannel: "); print(obj.nbrConvertersPerChannel()); printf("\n");
   indent(); printf("channelMask: "); print(obj.channelMask()); printf("\n");
   indent(); printf("nbrBanks: "); print(obj.nbrBanks()); printf("\n");
   indent(); printf("trig: "); print(obj.trig()); printf("\n");
   indent(); printf("horiz: "); print(obj.horiz()); printf("\n");

 
  {
  }
  INDENT -= 2;
}

void print(const Acqiris::DataDescV1 &obj) {
  printf("Acqiris DataDescV1\n");
  indent();
  INDENT += 2;

 
  INDENT -= 2;
}

void print(const Acqiris::DataDescV1Elem &obj) {
  printf("Acqiris DataDescV1Elem\n");
  indent();
  INDENT += 2;
   indent(); printf("nbrSamplesInSeg: "); print(obj.nbrSamplesInSeg()); printf("\n");
   indent(); printf("indexFirstPoint: "); print(obj.indexFirstPoint()); printf("\n");
   indent(); printf("nbrSegments: "); print(obj.nbrSegments()); printf("\n");

 
  {
  }
 
  {
    const int16_t *dataPtr = obj.waveforms().data();
    size_t numElements = obj.waveforms().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("waveforms (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Acqiris::HorizV1 &obj) {
  printf("Acqiris HorizV1\n");
  indent();
  INDENT += 2;
   indent(); printf("sampInterval: "); print(obj.sampInterval()); printf("\n");
   indent(); printf("delayTime: "); print(obj.delayTime()); printf("\n");
   indent(); printf("nbrSamples: "); print(obj.nbrSamples()); printf("\n");
   indent(); printf("nbrSegments: "); print(obj.nbrSegments()); printf("\n");

  INDENT -= 2;
}

void print(const Acqiris::TdcAuxIO &obj) {
  printf("Acqiris TdcAuxIO\n");
  indent();
  INDENT += 2;
   indent(); printf("channel: "); print(obj.channel()); printf("\n");
   indent(); printf("mode: "); print(obj.mode()); printf("\n");
   indent(); printf("term: "); print(obj.term()); printf("\n");

  INDENT -= 2;
}

void print(const Acqiris::TdcChannel &obj) {
  printf("Acqiris TdcChannel\n");
  indent();
  INDENT += 2;
   indent(); printf("channel: "); print(obj.channel()); printf("\n");
   indent(); printf("_mode_int: "); print(obj._mode_int()); printf("\n");
   indent(); printf("level: "); print(obj.level()); printf("\n");

  INDENT -= 2;
}

void print(const Acqiris::TdcConfigV1 &obj) {
  printf("Acqiris TdcConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("veto: "); print(obj.veto()); printf("\n");

 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const Acqiris::TdcDataV1 &obj) {
  printf("Acqiris TdcDataV1\n");
  indent();
  INDENT += 2;

 
  {
  }
  INDENT -= 2;
}

void print(const Acqiris::TdcDataV1Channel &obj) {
  printf("Acqiris TdcDataV1Channel\n");
  indent();
  INDENT += 2;

  INDENT -= 2;
}

void print(const Acqiris::TdcDataV1Common &obj) {
  printf("Acqiris TdcDataV1Common\n");
  indent();
  INDENT += 2;

  INDENT -= 2;
}

void print(const Acqiris::TdcDataV1Marker &obj) {
  printf("Acqiris TdcDataV1Marker\n");
  indent();
  INDENT += 2;

  INDENT -= 2;
}

void print(const Acqiris::TdcDataV1_Item &obj) {
  printf("Acqiris TdcDataV1_Item\n");
  indent();
  INDENT += 2;
   indent(); printf("value: "); print(obj.value()); printf("\n");

  INDENT -= 2;
}

void print(const Acqiris::TdcVetoIO &obj) {
  printf("Acqiris TdcVetoIO\n");
  indent();
  INDENT += 2;
   indent(); printf("channel: "); print(obj.channel()); printf("\n");
   indent(); printf("mode: "); print(obj.mode()); printf("\n");
   indent(); printf("term: "); print(obj.term()); printf("\n");

  INDENT -= 2;
}

void print(const Acqiris::TimestampV1 &obj) {
  printf("Acqiris TimestampV1\n");
  indent();
  INDENT += 2;
   indent(); printf("pos: "); print(obj.pos()); printf("\n");
   indent(); printf("timeStampLo: "); print(obj.timeStampLo()); printf("\n");
   indent(); printf("timeStampHi: "); print(obj.timeStampHi()); printf("\n");

  INDENT -= 2;
}

void print(const Acqiris::TrigV1 &obj) {
  printf("Acqiris TrigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("coupling: "); print(obj.coupling()); printf("\n");
   indent(); printf("input: "); print(obj.input()); printf("\n");
   indent(); printf("slope: "); print(obj.slope()); printf("\n");
   indent(); printf("level: "); print(obj.level()); printf("\n");

  INDENT -= 2;
}

void print(const Acqiris::VertV1 &obj) {
  printf("Acqiris VertV1\n");
  indent();
  INDENT += 2;
   indent(); printf("fullScale: "); print(obj.fullScale()); printf("\n");
   indent(); printf("offset: "); print(obj.offset()); printf("\n");
   indent(); printf("coupling: "); print(obj.coupling()); printf("\n");
   indent(); printf("bandwidth: "); print(obj.bandwidth()); printf("\n");

  INDENT -= 2;
}

void print(const Alias::ConfigV1 &obj) {
  printf("Alias ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("numSrcAlias: "); print(obj.numSrcAlias()); printf("\n");

 
  {
  }
  INDENT -= 2;
}

void print(const Alias::SrcAlias &obj) {
  printf("Alias SrcAlias\n");
  indent();
  INDENT += 2;
   indent(); printf("src: "); print(obj.src()); printf("\n");

 
  printf(" aliasName: %s\n", obj.aliasName());
  INDENT -= 2;
}

void print(const Andor::ConfigV1 &obj) {
  printf("Andor ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("width: "); print(obj.width()); printf("\n");
   indent(); printf("height: "); print(obj.height()); printf("\n");
   indent(); printf("orgX: "); print(obj.orgX()); printf("\n");
   indent(); printf("orgY: "); print(obj.orgY()); printf("\n");
   indent(); printf("binX: "); print(obj.binX()); printf("\n");
   indent(); printf("binY: "); print(obj.binY()); printf("\n");
   indent(); printf("exposureTime: "); print(obj.exposureTime()); printf("\n");
   indent(); printf("coolingTemp: "); print(obj.coolingTemp()); printf("\n");
   indent(); printf("fanMode: "); print(obj.fanMode()); printf("\n");
   indent(); printf("baselineClamp: "); print(obj.baselineClamp()); printf("\n");
   indent(); printf("highCapacity: "); print(obj.highCapacity()); printf("\n");
   indent(); printf("gainIndex: "); print(obj.gainIndex()); printf("\n");
   indent(); printf("readoutSpeedIndex: "); print(obj.readoutSpeedIndex()); printf("\n");
   indent(); printf("exposureEventCode: "); print(obj.exposureEventCode()); printf("\n");
   indent(); printf("numDelayShots: "); print(obj.numDelayShots()); printf("\n");

  INDENT -= 2;
}

void print(const Andor::FrameV1 &obj) {
  printf("Andor FrameV1\n");
  indent();
  INDENT += 2;
   indent(); printf("shotIdStart: "); print(obj.shotIdStart()); printf("\n");
   indent(); printf("readoutTime: "); print(obj.readoutTime()); printf("\n");
   indent(); printf("temperature: "); print(obj.temperature()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Bld::BldDataAcqADCV1 &obj) {
  printf("Bld BldDataAcqADCV1\n");
  indent();
  INDENT += 2;
   indent(); printf("config: "); print(obj.config()); printf("\n");
   indent(); printf("data: "); print(obj.data()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataEBeamV0 &obj) {
  printf("Bld BldDataEBeamV0\n");
  indent();
  INDENT += 2;
   indent(); printf("damageMask: "); print(obj.damageMask()); printf("\n");
   indent(); printf("ebeamCharge: "); print(obj.ebeamCharge()); printf("\n");
   indent(); printf("ebeamL3Energy: "); print(obj.ebeamL3Energy()); printf("\n");
   indent(); printf("ebeamLTUPosX: "); print(obj.ebeamLTUPosX()); printf("\n");
   indent(); printf("ebeamLTUPosY: "); print(obj.ebeamLTUPosY()); printf("\n");
   indent(); printf("ebeamLTUAngX: "); print(obj.ebeamLTUAngX()); printf("\n");
   indent(); printf("ebeamLTUAngY: "); print(obj.ebeamLTUAngY()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataEBeamV1 &obj) {
  printf("Bld BldDataEBeamV1\n");
  indent();
  INDENT += 2;
   indent(); printf("damageMask: "); print(obj.damageMask()); printf("\n");
   indent(); printf("ebeamCharge: "); print(obj.ebeamCharge()); printf("\n");
   indent(); printf("ebeamL3Energy: "); print(obj.ebeamL3Energy()); printf("\n");
   indent(); printf("ebeamLTUPosX: "); print(obj.ebeamLTUPosX()); printf("\n");
   indent(); printf("ebeamLTUPosY: "); print(obj.ebeamLTUPosY()); printf("\n");
   indent(); printf("ebeamLTUAngX: "); print(obj.ebeamLTUAngX()); printf("\n");
   indent(); printf("ebeamLTUAngY: "); print(obj.ebeamLTUAngY()); printf("\n");
   indent(); printf("ebeamPkCurrBC2: "); print(obj.ebeamPkCurrBC2()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataEBeamV2 &obj) {
  printf("Bld BldDataEBeamV2\n");
  indent();
  INDENT += 2;
   indent(); printf("damageMask: "); print(obj.damageMask()); printf("\n");
   indent(); printf("ebeamCharge: "); print(obj.ebeamCharge()); printf("\n");
   indent(); printf("ebeamL3Energy: "); print(obj.ebeamL3Energy()); printf("\n");
   indent(); printf("ebeamLTUPosX: "); print(obj.ebeamLTUPosX()); printf("\n");
   indent(); printf("ebeamLTUPosY: "); print(obj.ebeamLTUPosY()); printf("\n");
   indent(); printf("ebeamLTUAngX: "); print(obj.ebeamLTUAngX()); printf("\n");
   indent(); printf("ebeamLTUAngY: "); print(obj.ebeamLTUAngY()); printf("\n");
   indent(); printf("ebeamPkCurrBC2: "); print(obj.ebeamPkCurrBC2()); printf("\n");
   indent(); printf("ebeamEnergyBC2: "); print(obj.ebeamEnergyBC2()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataEBeamV3 &obj) {
  printf("Bld BldDataEBeamV3\n");
  indent();
  INDENT += 2;
   indent(); printf("damageMask: "); print(obj.damageMask()); printf("\n");
   indent(); printf("ebeamCharge: "); print(obj.ebeamCharge()); printf("\n");
   indent(); printf("ebeamL3Energy: "); print(obj.ebeamL3Energy()); printf("\n");
   indent(); printf("ebeamLTUPosX: "); print(obj.ebeamLTUPosX()); printf("\n");
   indent(); printf("ebeamLTUPosY: "); print(obj.ebeamLTUPosY()); printf("\n");
   indent(); printf("ebeamLTUAngX: "); print(obj.ebeamLTUAngX()); printf("\n");
   indent(); printf("ebeamLTUAngY: "); print(obj.ebeamLTUAngY()); printf("\n");
   indent(); printf("ebeamPkCurrBC2: "); print(obj.ebeamPkCurrBC2()); printf("\n");
   indent(); printf("ebeamEnergyBC2: "); print(obj.ebeamEnergyBC2()); printf("\n");
   indent(); printf("ebeamPkCurrBC1: "); print(obj.ebeamPkCurrBC1()); printf("\n");
   indent(); printf("ebeamEnergyBC1: "); print(obj.ebeamEnergyBC1()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataEBeamV4 &obj) {
  printf("Bld BldDataEBeamV4\n");
  indent();
  INDENT += 2;
   indent(); printf("damageMask: "); print(obj.damageMask()); printf("\n");
   indent(); printf("ebeamCharge: "); print(obj.ebeamCharge()); printf("\n");
   indent(); printf("ebeamL3Energy: "); print(obj.ebeamL3Energy()); printf("\n");
   indent(); printf("ebeamLTUPosX: "); print(obj.ebeamLTUPosX()); printf("\n");
   indent(); printf("ebeamLTUPosY: "); print(obj.ebeamLTUPosY()); printf("\n");
   indent(); printf("ebeamLTUAngX: "); print(obj.ebeamLTUAngX()); printf("\n");
   indent(); printf("ebeamLTUAngY: "); print(obj.ebeamLTUAngY()); printf("\n");
   indent(); printf("ebeamPkCurrBC2: "); print(obj.ebeamPkCurrBC2()); printf("\n");
   indent(); printf("ebeamEnergyBC2: "); print(obj.ebeamEnergyBC2()); printf("\n");
   indent(); printf("ebeamPkCurrBC1: "); print(obj.ebeamPkCurrBC1()); printf("\n");
   indent(); printf("ebeamEnergyBC1: "); print(obj.ebeamEnergyBC1()); printf("\n");
   indent(); printf("ebeamUndPosX: "); print(obj.ebeamUndPosX()); printf("\n");
   indent(); printf("ebeamUndPosY: "); print(obj.ebeamUndPosY()); printf("\n");
   indent(); printf("ebeamUndAngX: "); print(obj.ebeamUndAngX()); printf("\n");
   indent(); printf("ebeamUndAngY: "); print(obj.ebeamUndAngY()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataFEEGasDetEnergy &obj) {
  printf("Bld BldDataFEEGasDetEnergy\n");
  indent();
  INDENT += 2;
   indent(); printf("f_11_ENRC: "); print(obj.f_11_ENRC()); printf("\n");
   indent(); printf("f_12_ENRC: "); print(obj.f_12_ENRC()); printf("\n");
   indent(); printf("f_21_ENRC: "); print(obj.f_21_ENRC()); printf("\n");
   indent(); printf("f_22_ENRC: "); print(obj.f_22_ENRC()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataGMDV0 &obj) {
  printf("Bld BldDataGMDV0\n");
  indent();
  INDENT += 2;
   indent(); printf("pressure: "); print(obj.pressure()); printf("\n");
   indent(); printf("temperature: "); print(obj.temperature()); printf("\n");
   indent(); printf("current: "); print(obj.current()); printf("\n");
   indent(); printf("hvMeshElectron: "); print(obj.hvMeshElectron()); printf("\n");
   indent(); printf("hvMeshIon: "); print(obj.hvMeshIon()); printf("\n");
   indent(); printf("hvMultIon: "); print(obj.hvMultIon()); printf("\n");
   indent(); printf("chargeQ: "); print(obj.chargeQ()); printf("\n");
   indent(); printf("photonEnergy: "); print(obj.photonEnergy()); printf("\n");
   indent(); printf("multPulseIntensity: "); print(obj.multPulseIntensity()); printf("\n");
   indent(); printf("keithleyPulseIntensity: "); print(obj.keithleyPulseIntensity()); printf("\n");
   indent(); printf("pulseEnergy: "); print(obj.pulseEnergy()); printf("\n");
   indent(); printf("pulseEnergyFEE: "); print(obj.pulseEnergyFEE()); printf("\n");
   indent(); printf("transmission: "); print(obj.transmission()); printf("\n");
   indent(); printf("transmissionFEE: "); print(obj.transmissionFEE()); printf("\n");

 
  printf(" gasType: %s\n", obj.gasType());
  INDENT -= 2;
}

void print(const Bld::BldDataGMDV1 &obj) {
  printf("Bld BldDataGMDV1\n");
  indent();
  INDENT += 2;
   indent(); printf("milliJoulesPerPulse: "); print(obj.milliJoulesPerPulse()); printf("\n");
   indent(); printf("milliJoulesAverage: "); print(obj.milliJoulesAverage()); printf("\n");
   indent(); printf("correctedSumPerPulse: "); print(obj.correctedSumPerPulse()); printf("\n");
   indent(); printf("bgValuePerSample: "); print(obj.bgValuePerSample()); printf("\n");
   indent(); printf("relativeEnergyPerPulse: "); print(obj.relativeEnergyPerPulse()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataIpimbV0 &obj) {
  printf("Bld BldDataIpimbV0\n");
  indent();
  INDENT += 2;
   indent(); printf("ipimbData: "); print(obj.ipimbData()); printf("\n");
   indent(); printf("ipimbConfig: "); print(obj.ipimbConfig()); printf("\n");
   indent(); printf("ipmFexData: "); print(obj.ipmFexData()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataIpimbV1 &obj) {
  printf("Bld BldDataIpimbV1\n");
  indent();
  INDENT += 2;
   indent(); printf("ipimbData: "); print(obj.ipimbData()); printf("\n");
   indent(); printf("ipimbConfig: "); print(obj.ipimbConfig()); printf("\n");
   indent(); printf("ipmFexData: "); print(obj.ipmFexData()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataPhaseCavity &obj) {
  printf("Bld BldDataPhaseCavity\n");
  indent();
  INDENT += 2;
   indent(); printf("fitTime1: "); print(obj.fitTime1()); printf("\n");
   indent(); printf("fitTime2: "); print(obj.fitTime2()); printf("\n");
   indent(); printf("charge1: "); print(obj.charge1()); printf("\n");
   indent(); printf("charge2: "); print(obj.charge2()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataPimV1 &obj) {
  printf("Bld BldDataPimV1\n");
  indent();
  INDENT += 2;
   indent(); printf("camConfig: "); print(obj.camConfig()); printf("\n");
   indent(); printf("pimConfig: "); print(obj.pimConfig()); printf("\n");
   indent(); printf("frame: "); print(obj.frame()); printf("\n");

  INDENT -= 2;
}

void print(const Bld::BldDataSpectrometerV0 &obj) {
  printf("Bld BldDataSpectrometerV0\n");
  indent();
  INDENT += 2;

 
  {
    const uint32_t *dataPtr = obj.hproj().data();
    size_t numElements = obj.hproj().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("hproj (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint32_t *dataPtr = obj.vproj().data();
    size_t numElements = obj.vproj().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("vproj (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Camera::FrameCoord &obj) {
  printf("Camera FrameCoord\n");
  indent();
  INDENT += 2;
   indent(); printf("column: "); print(obj.column()); printf("\n");
   indent(); printf("row: "); print(obj.row()); printf("\n");

  INDENT -= 2;
}

void print(const Camera::FrameFccdConfigV1 &obj) {
  printf("Camera FrameFccdConfigV1\n");
  indent();
  INDENT += 2;

  INDENT -= 2;
}

void print(const Camera::FrameFexConfigV1 &obj) {
  printf("Camera FrameFexConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("forwarding: "); print(obj.forwarding()); printf("\n");
   indent(); printf("forward_prescale: "); print(obj.forward_prescale()); printf("\n");
   indent(); printf("processing: "); print(obj.processing()); printf("\n");
   indent(); printf("roiBegin: "); print(obj.roiBegin()); printf("\n");
   indent(); printf("roiEnd: "); print(obj.roiEnd()); printf("\n");
   indent(); printf("threshold: "); print(obj.threshold()); printf("\n");
   indent(); printf("number_of_masked_pixels: "); print(obj.number_of_masked_pixels()); printf("\n");

 
  {
  }
  INDENT -= 2;
}

void print(const Camera::FrameV1 &obj) {
  printf("Camera FrameV1\n");
  indent();
  INDENT += 2;
   indent(); printf("width: "); print(obj.width()); printf("\n");
   indent(); printf("height: "); print(obj.height()); printf("\n");
   indent(); printf("depth: "); print(obj.depth()); printf("\n");
   indent(); printf("offset: "); print(obj.offset()); printf("\n");

 
  {
    const uint8_t *dataPtr = obj._int_pixel_data().data();
    size_t numElements = obj._int_pixel_data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("_int_pixel_data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Camera::TwoDGaussianV1 &obj) {
  printf("Camera TwoDGaussianV1\n");
  indent();
  INDENT += 2;
   indent(); printf("integral: "); print(obj.integral()); printf("\n");
   indent(); printf("xmean: "); print(obj.xmean()); printf("\n");
   indent(); printf("ymean: "); print(obj.ymean()); printf("\n");
   indent(); printf("major_axis_width: "); print(obj.major_axis_width()); printf("\n");
   indent(); printf("minor_axis_width: "); print(obj.minor_axis_width()); printf("\n");
   indent(); printf("major_axis_tilt: "); print(obj.major_axis_tilt()); printf("\n");

  INDENT -= 2;
}

void print(const ControlData::ConfigV1 &obj) {
  printf("ControlData ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("duration: "); print(obj.duration()); printf("\n");
   indent(); printf("npvControls: "); print(obj.npvControls()); printf("\n");
   indent(); printf("npvMonitors: "); print(obj.npvMonitors()); printf("\n");

 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const ControlData::ConfigV2 &obj) {
  printf("ControlData ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("duration: "); print(obj.duration()); printf("\n");
   indent(); printf("npvControls: "); print(obj.npvControls()); printf("\n");
   indent(); printf("npvMonitors: "); print(obj.npvMonitors()); printf("\n");
   indent(); printf("npvLabels: "); print(obj.npvLabels()); printf("\n");

 
  {
  }
 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const ControlData::ConfigV3 &obj) {
  printf("ControlData ConfigV3\n");
  indent();
  INDENT += 2;
   indent(); printf("duration: "); print(obj.duration()); printf("\n");
   indent(); printf("npvControls: "); print(obj.npvControls()); printf("\n");
   indent(); printf("npvMonitors: "); print(obj.npvMonitors()); printf("\n");
   indent(); printf("npvLabels: "); print(obj.npvLabels()); printf("\n");

 
  {
  }
 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const ControlData::PVControl &obj) {
  printf("ControlData PVControl\n");
  indent();
  INDENT += 2;
   indent(); printf("index: "); print(obj.index()); printf("\n");
   indent(); printf("value: "); print(obj.value()); printf("\n");

 
  printf(" name: %s\n", obj.name());
  INDENT -= 2;
}

void print(const ControlData::PVLabel &obj) {
  printf("ControlData PVLabel\n");
  indent();
  INDENT += 2;

 
  printf(" name: %s\n", obj.name());
 
  printf(" value: %s\n", obj.value());
  INDENT -= 2;
}

void print(const ControlData::PVMonitor &obj) {
  printf("ControlData PVMonitor\n");
  indent();
  INDENT += 2;
   indent(); printf("index: "); print(obj.index()); printf("\n");
   indent(); printf("loValue: "); print(obj.loValue()); printf("\n");
   indent(); printf("hiValue: "); print(obj.hiValue()); printf("\n");

 
  printf(" name: %s\n", obj.name());
  INDENT -= 2;
}

void print(const CsPad::ConfigV1 &obj) {
  printf("CsPad ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("concentratorVersion: "); print(obj.concentratorVersion()); printf("\n");
   indent(); printf("runDelay: "); print(obj.runDelay()); printf("\n");
   indent(); printf("eventCode: "); print(obj.eventCode()); printf("\n");
   indent(); printf("inactiveRunMode: "); print(obj.inactiveRunMode()); printf("\n");
   indent(); printf("activeRunMode: "); print(obj.activeRunMode()); printf("\n");
   indent(); printf("tdi: "); print(obj.tdi()); printf("\n");
   indent(); printf("payloadSize: "); print(obj.payloadSize()); printf("\n");
   indent(); printf("badAsicMask0: "); print(obj.badAsicMask0()); printf("\n");
   indent(); printf("badAsicMask1: "); print(obj.badAsicMask1()); printf("\n");
   indent(); printf("asicMask: "); print(obj.asicMask()); printf("\n");
   indent(); printf("quadMask: "); print(obj.quadMask()); printf("\n");

 
  INDENT -= 2;
}

void print(const CsPad::ConfigV1QuadReg &obj) {
  printf("CsPad ConfigV1QuadReg\n");
  indent();
  INDENT += 2;
   indent(); printf("readClkSet: "); print(obj.readClkSet()); printf("\n");
   indent(); printf("readClkHold: "); print(obj.readClkHold()); printf("\n");
   indent(); printf("dataMode: "); print(obj.dataMode()); printf("\n");
   indent(); printf("prstSel: "); print(obj.prstSel()); printf("\n");
   indent(); printf("acqDelay: "); print(obj.acqDelay()); printf("\n");
   indent(); printf("intTime: "); print(obj.intTime()); printf("\n");
   indent(); printf("digDelay: "); print(obj.digDelay()); printf("\n");
   indent(); printf("ampIdle: "); print(obj.ampIdle()); printf("\n");
   indent(); printf("injTotal: "); print(obj.injTotal()); printf("\n");
   indent(); printf("rowColShiftPer: "); print(obj.rowColShiftPer()); printf("\n");
   indent(); printf("ro: "); print(obj.ro()); printf("\n");
   indent(); printf("dp: "); print(obj.dp()); printf("\n");
   indent(); printf("gm: "); print(obj.gm()); printf("\n");

 
  {
    const uint32_t *dataPtr = obj.shiftSelect().data();
    size_t numElements = obj.shiftSelect().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("shiftSelect (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint32_t *dataPtr = obj.edgeSelect().data();
    size_t numElements = obj.edgeSelect().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("edgeSelect (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad::ConfigV2 &obj) {
  printf("CsPad ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("concentratorVersion: "); print(obj.concentratorVersion()); printf("\n");
   indent(); printf("runDelay: "); print(obj.runDelay()); printf("\n");
   indent(); printf("eventCode: "); print(obj.eventCode()); printf("\n");
   indent(); printf("inactiveRunMode: "); print(obj.inactiveRunMode()); printf("\n");
   indent(); printf("activeRunMode: "); print(obj.activeRunMode()); printf("\n");
   indent(); printf("tdi: "); print(obj.tdi()); printf("\n");
   indent(); printf("payloadSize: "); print(obj.payloadSize()); printf("\n");
   indent(); printf("badAsicMask0: "); print(obj.badAsicMask0()); printf("\n");
   indent(); printf("badAsicMask1: "); print(obj.badAsicMask1()); printf("\n");
   indent(); printf("asicMask: "); print(obj.asicMask()); printf("\n");
   indent(); printf("quadMask: "); print(obj.quadMask()); printf("\n");
   indent(); printf("roiMasks: "); print(obj.roiMasks()); printf("\n");

 
  INDENT -= 2;
}

void print(const CsPad::ConfigV2QuadReg &obj) {
  printf("CsPad ConfigV2QuadReg\n");
  indent();
  INDENT += 2;
   indent(); printf("readClkSet: "); print(obj.readClkSet()); printf("\n");
   indent(); printf("readClkHold: "); print(obj.readClkHold()); printf("\n");
   indent(); printf("dataMode: "); print(obj.dataMode()); printf("\n");
   indent(); printf("prstSel: "); print(obj.prstSel()); printf("\n");
   indent(); printf("acqDelay: "); print(obj.acqDelay()); printf("\n");
   indent(); printf("intTime: "); print(obj.intTime()); printf("\n");
   indent(); printf("digDelay: "); print(obj.digDelay()); printf("\n");
   indent(); printf("ampIdle: "); print(obj.ampIdle()); printf("\n");
   indent(); printf("injTotal: "); print(obj.injTotal()); printf("\n");
   indent(); printf("rowColShiftPer: "); print(obj.rowColShiftPer()); printf("\n");
   indent(); printf("ampReset: "); print(obj.ampReset()); printf("\n");
   indent(); printf("digCount: "); print(obj.digCount()); printf("\n");
   indent(); printf("digPeriod: "); print(obj.digPeriod()); printf("\n");
   indent(); printf("ro: "); print(obj.ro()); printf("\n");
   indent(); printf("dp: "); print(obj.dp()); printf("\n");
   indent(); printf("gm: "); print(obj.gm()); printf("\n");

 
  {
    const uint32_t *dataPtr = obj.shiftSelect().data();
    size_t numElements = obj.shiftSelect().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("shiftSelect (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint32_t *dataPtr = obj.edgeSelect().data();
    size_t numElements = obj.edgeSelect().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("edgeSelect (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad::ConfigV3 &obj) {
  printf("CsPad ConfigV3\n");
  indent();
  INDENT += 2;
   indent(); printf("concentratorVersion: "); print(obj.concentratorVersion()); printf("\n");
   indent(); printf("runDelay: "); print(obj.runDelay()); printf("\n");
   indent(); printf("eventCode: "); print(obj.eventCode()); printf("\n");
   indent(); printf("protectionEnable: "); print(obj.protectionEnable()); printf("\n");
   indent(); printf("inactiveRunMode: "); print(obj.inactiveRunMode()); printf("\n");
   indent(); printf("activeRunMode: "); print(obj.activeRunMode()); printf("\n");
   indent(); printf("tdi: "); print(obj.tdi()); printf("\n");
   indent(); printf("payloadSize: "); print(obj.payloadSize()); printf("\n");
   indent(); printf("badAsicMask0: "); print(obj.badAsicMask0()); printf("\n");
   indent(); printf("badAsicMask1: "); print(obj.badAsicMask1()); printf("\n");
   indent(); printf("asicMask: "); print(obj.asicMask()); printf("\n");
   indent(); printf("quadMask: "); print(obj.quadMask()); printf("\n");
   indent(); printf("roiMasks: "); print(obj.roiMasks()); printf("\n");

 
  {
  }
 
  INDENT -= 2;
}

void print(const CsPad::ConfigV3QuadReg &obj) {
  printf("CsPad ConfigV3QuadReg\n");
  indent();
  INDENT += 2;
   indent(); printf("readClkSet: "); print(obj.readClkSet()); printf("\n");
   indent(); printf("readClkHold: "); print(obj.readClkHold()); printf("\n");
   indent(); printf("dataMode: "); print(obj.dataMode()); printf("\n");
   indent(); printf("prstSel: "); print(obj.prstSel()); printf("\n");
   indent(); printf("acqDelay: "); print(obj.acqDelay()); printf("\n");
   indent(); printf("intTime: "); print(obj.intTime()); printf("\n");
   indent(); printf("digDelay: "); print(obj.digDelay()); printf("\n");
   indent(); printf("ampIdle: "); print(obj.ampIdle()); printf("\n");
   indent(); printf("injTotal: "); print(obj.injTotal()); printf("\n");
   indent(); printf("rowColShiftPer: "); print(obj.rowColShiftPer()); printf("\n");
   indent(); printf("ampReset: "); print(obj.ampReset()); printf("\n");
   indent(); printf("digCount: "); print(obj.digCount()); printf("\n");
   indent(); printf("digPeriod: "); print(obj.digPeriod()); printf("\n");
   indent(); printf("biasTuning: "); print(obj.biasTuning()); printf("\n");
   indent(); printf("pdpmndnmBalance: "); print(obj.pdpmndnmBalance()); printf("\n");
   indent(); printf("ro: "); print(obj.ro()); printf("\n");
   indent(); printf("dp: "); print(obj.dp()); printf("\n");
   indent(); printf("gm: "); print(obj.gm()); printf("\n");

 
  {
    const uint32_t *dataPtr = obj.shiftSelect().data();
    size_t numElements = obj.shiftSelect().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("shiftSelect (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint32_t *dataPtr = obj.edgeSelect().data();
    size_t numElements = obj.edgeSelect().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("edgeSelect (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad::ConfigV4 &obj) {
  printf("CsPad ConfigV4\n");
  indent();
  INDENT += 2;
   indent(); printf("concentratorVersion: "); print(obj.concentratorVersion()); printf("\n");
   indent(); printf("runDelay: "); print(obj.runDelay()); printf("\n");
   indent(); printf("eventCode: "); print(obj.eventCode()); printf("\n");
   indent(); printf("protectionEnable: "); print(obj.protectionEnable()); printf("\n");
   indent(); printf("inactiveRunMode: "); print(obj.inactiveRunMode()); printf("\n");
   indent(); printf("activeRunMode: "); print(obj.activeRunMode()); printf("\n");
   indent(); printf("tdi: "); print(obj.tdi()); printf("\n");
   indent(); printf("payloadSize: "); print(obj.payloadSize()); printf("\n");
   indent(); printf("badAsicMask0: "); print(obj.badAsicMask0()); printf("\n");
   indent(); printf("badAsicMask1: "); print(obj.badAsicMask1()); printf("\n");
   indent(); printf("asicMask: "); print(obj.asicMask()); printf("\n");
   indent(); printf("quadMask: "); print(obj.quadMask()); printf("\n");
   indent(); printf("roiMasks: "); print(obj.roiMasks()); printf("\n");

 
  {
  }
 
  INDENT -= 2;
}

void print(const CsPad::ConfigV5 &obj) {
  printf("CsPad ConfigV5\n");
  indent();
  INDENT += 2;
   indent(); printf("concentratorVersion: "); print(obj.concentratorVersion()); printf("\n");
   indent(); printf("runDelay: "); print(obj.runDelay()); printf("\n");
   indent(); printf("eventCode: "); print(obj.eventCode()); printf("\n");
   indent(); printf("protectionEnable: "); print(obj.protectionEnable()); printf("\n");
   indent(); printf("inactiveRunMode: "); print(obj.inactiveRunMode()); printf("\n");
   indent(); printf("activeRunMode: "); print(obj.activeRunMode()); printf("\n");
   indent(); printf("internalTriggerDelay: "); print(obj.internalTriggerDelay()); printf("\n");
   indent(); printf("tdi: "); print(obj.tdi()); printf("\n");
   indent(); printf("payloadSize: "); print(obj.payloadSize()); printf("\n");
   indent(); printf("badAsicMask0: "); print(obj.badAsicMask0()); printf("\n");
   indent(); printf("badAsicMask1: "); print(obj.badAsicMask1()); printf("\n");
   indent(); printf("asicMask: "); print(obj.asicMask()); printf("\n");
   indent(); printf("quadMask: "); print(obj.quadMask()); printf("\n");
   indent(); printf("roiMasks: "); print(obj.roiMasks()); printf("\n");

 
  {
  }
 
  INDENT -= 2;
}

void print(const CsPad::CsPadDigitalPotsCfg &obj) {
  printf("CsPad CsPadDigitalPotsCfg\n");
  indent();
  INDENT += 2;

 
  {
    const uint8_t *dataPtr = obj.pots().data();
    size_t numElements = obj.pots().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("pots (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad::CsPadGainMapCfg &obj) {
  printf("CsPad CsPadGainMapCfg\n");
  indent();
  INDENT += 2;

 
  {
    const uint16_t *dataPtr = obj.gainMap().data();
    size_t numElements = obj.gainMap().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("gainMap (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad::CsPadReadOnlyCfg &obj) {
  printf("CsPad CsPadReadOnlyCfg\n");
  indent();
  INDENT += 2;
   indent(); printf("shiftTest: "); print(obj.shiftTest()); printf("\n");
   indent(); printf("version: "); print(obj.version()); printf("\n");

  INDENT -= 2;
}

void print(const CsPad::DataV1 &obj) {
  printf("CsPad DataV1\n");
  indent();
  INDENT += 2;

 
  INDENT -= 2;
}

void print(const CsPad::DataV2 &obj) {
  printf("CsPad DataV2\n");
  indent();
  INDENT += 2;

 
  INDENT -= 2;
}

void print(const CsPad::ElementV1 &obj) {
  printf("CsPad ElementV1\n");
  indent();
  INDENT += 2;
   indent(); printf("seq_count: "); print(obj.seq_count()); printf("\n");
   indent(); printf("ticks: "); print(obj.ticks()); printf("\n");
   indent(); printf("fiducials: "); print(obj.fiducials()); printf("\n");
   indent(); printf("frame_type: "); print(obj.frame_type()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.sb_temp().data();
    size_t numElements = obj.sb_temp().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("sb_temp (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const int16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad::ElementV2 &obj) {
  printf("CsPad ElementV2\n");
  indent();
  INDENT += 2;
   indent(); printf("seq_count: "); print(obj.seq_count()); printf("\n");
   indent(); printf("ticks: "); print(obj.ticks()); printf("\n");
   indent(); printf("fiducials: "); print(obj.fiducials()); printf("\n");
   indent(); printf("frame_type: "); print(obj.frame_type()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.sb_temp().data();
    size_t numElements = obj.sb_temp().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("sb_temp (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const int16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad::ProtectionSystemThreshold &obj) {
  printf("CsPad ProtectionSystemThreshold\n");
  indent();
  INDENT += 2;
   indent(); printf("adcThreshold: "); print(obj.adcThreshold()); printf("\n");
   indent(); printf("pixelCountThreshold: "); print(obj.pixelCountThreshold()); printf("\n");

  INDENT -= 2;
}

void print(const CsPad2x2::ConfigV1 &obj) {
  printf("CsPad2x2 ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("concentratorVersion: "); print(obj.concentratorVersion()); printf("\n");
   indent(); printf("protectionThreshold: "); print(obj.protectionThreshold()); printf("\n");
   indent(); printf("protectionEnable: "); print(obj.protectionEnable()); printf("\n");
   indent(); printf("inactiveRunMode: "); print(obj.inactiveRunMode()); printf("\n");
   indent(); printf("activeRunMode: "); print(obj.activeRunMode()); printf("\n");
   indent(); printf("tdi: "); print(obj.tdi()); printf("\n");
   indent(); printf("payloadSize: "); print(obj.payloadSize()); printf("\n");
   indent(); printf("badAsicMask: "); print(obj.badAsicMask()); printf("\n");
   indent(); printf("asicMask: "); print(obj.asicMask()); printf("\n");
   indent(); printf("roiMask: "); print(obj.roiMask()); printf("\n");
   indent(); printf("quad: "); print(obj.quad()); printf("\n");

  INDENT -= 2;
}

void print(const CsPad2x2::ConfigV1QuadReg &obj) {
  printf("CsPad2x2 ConfigV1QuadReg\n");
  indent();
  INDENT += 2;
   indent(); printf("shiftSelect: "); print(obj.shiftSelect()); printf("\n");
   indent(); printf("edgeSelect: "); print(obj.edgeSelect()); printf("\n");
   indent(); printf("readClkSet: "); print(obj.readClkSet()); printf("\n");
   indent(); printf("readClkHold: "); print(obj.readClkHold()); printf("\n");
   indent(); printf("dataMode: "); print(obj.dataMode()); printf("\n");
   indent(); printf("prstSel: "); print(obj.prstSel()); printf("\n");
   indent(); printf("acqDelay: "); print(obj.acqDelay()); printf("\n");
   indent(); printf("intTime: "); print(obj.intTime()); printf("\n");
   indent(); printf("digDelay: "); print(obj.digDelay()); printf("\n");
   indent(); printf("ampIdle: "); print(obj.ampIdle()); printf("\n");
   indent(); printf("injTotal: "); print(obj.injTotal()); printf("\n");
   indent(); printf("rowColShiftPer: "); print(obj.rowColShiftPer()); printf("\n");
   indent(); printf("ampReset: "); print(obj.ampReset()); printf("\n");
   indent(); printf("digCount: "); print(obj.digCount()); printf("\n");
   indent(); printf("digPeriod: "); print(obj.digPeriod()); printf("\n");
   indent(); printf("PeltierEnable: "); print(obj.PeltierEnable()); printf("\n");
   indent(); printf("kpConstant: "); print(obj.kpConstant()); printf("\n");
   indent(); printf("kiConstant: "); print(obj.kiConstant()); printf("\n");
   indent(); printf("kdConstant: "); print(obj.kdConstant()); printf("\n");
   indent(); printf("humidThold: "); print(obj.humidThold()); printf("\n");
   indent(); printf("setPoint: "); print(obj.setPoint()); printf("\n");
   indent(); printf("ro: "); print(obj.ro()); printf("\n");
   indent(); printf("dp: "); print(obj.dp()); printf("\n");
   indent(); printf("gm: "); print(obj.gm()); printf("\n");

  INDENT -= 2;
}

void print(const CsPad2x2::ConfigV2 &obj) {
  printf("CsPad2x2 ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("concentratorVersion: "); print(obj.concentratorVersion()); printf("\n");
   indent(); printf("protectionThreshold: "); print(obj.protectionThreshold()); printf("\n");
   indent(); printf("protectionEnable: "); print(obj.protectionEnable()); printf("\n");
   indent(); printf("inactiveRunMode: "); print(obj.inactiveRunMode()); printf("\n");
   indent(); printf("activeRunMode: "); print(obj.activeRunMode()); printf("\n");
   indent(); printf("runTriggerDelay: "); print(obj.runTriggerDelay()); printf("\n");
   indent(); printf("tdi: "); print(obj.tdi()); printf("\n");
   indent(); printf("payloadSize: "); print(obj.payloadSize()); printf("\n");
   indent(); printf("badAsicMask: "); print(obj.badAsicMask()); printf("\n");
   indent(); printf("asicMask: "); print(obj.asicMask()); printf("\n");
   indent(); printf("roiMask: "); print(obj.roiMask()); printf("\n");
   indent(); printf("quad: "); print(obj.quad()); printf("\n");

  INDENT -= 2;
}

void print(const CsPad2x2::ConfigV2QuadReg &obj) {
  printf("CsPad2x2 ConfigV2QuadReg\n");
  indent();
  INDENT += 2;
   indent(); printf("shiftSelect: "); print(obj.shiftSelect()); printf("\n");
   indent(); printf("edgeSelect: "); print(obj.edgeSelect()); printf("\n");
   indent(); printf("readClkSet: "); print(obj.readClkSet()); printf("\n");
   indent(); printf("readClkHold: "); print(obj.readClkHold()); printf("\n");
   indent(); printf("dataMode: "); print(obj.dataMode()); printf("\n");
   indent(); printf("prstSel: "); print(obj.prstSel()); printf("\n");
   indent(); printf("acqDelay: "); print(obj.acqDelay()); printf("\n");
   indent(); printf("intTime: "); print(obj.intTime()); printf("\n");
   indent(); printf("digDelay: "); print(obj.digDelay()); printf("\n");
   indent(); printf("ampIdle: "); print(obj.ampIdle()); printf("\n");
   indent(); printf("injTotal: "); print(obj.injTotal()); printf("\n");
   indent(); printf("rowColShiftPer: "); print(obj.rowColShiftPer()); printf("\n");
   indent(); printf("ampReset: "); print(obj.ampReset()); printf("\n");
   indent(); printf("digCount: "); print(obj.digCount()); printf("\n");
   indent(); printf("digPeriod: "); print(obj.digPeriod()); printf("\n");
   indent(); printf("PeltierEnable: "); print(obj.PeltierEnable()); printf("\n");
   indent(); printf("kpConstant: "); print(obj.kpConstant()); printf("\n");
   indent(); printf("kiConstant: "); print(obj.kiConstant()); printf("\n");
   indent(); printf("kdConstant: "); print(obj.kdConstant()); printf("\n");
   indent(); printf("humidThold: "); print(obj.humidThold()); printf("\n");
   indent(); printf("setPoint: "); print(obj.setPoint()); printf("\n");
   indent(); printf("biasTuning: "); print(obj.biasTuning()); printf("\n");
   indent(); printf("pdpmndnmBalance: "); print(obj.pdpmndnmBalance()); printf("\n");
   indent(); printf("ro: "); print(obj.ro()); printf("\n");
   indent(); printf("dp: "); print(obj.dp()); printf("\n");
   indent(); printf("gm: "); print(obj.gm()); printf("\n");

  INDENT -= 2;
}

void print(const CsPad2x2::CsPad2x2DigitalPotsCfg &obj) {
  printf("CsPad2x2 CsPad2x2DigitalPotsCfg\n");
  indent();
  INDENT += 2;

 
  {
    const uint8_t *dataPtr = obj.pots().data();
    size_t numElements = obj.pots().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("pots (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad2x2::CsPad2x2GainMapCfg &obj) {
  printf("CsPad2x2 CsPad2x2GainMapCfg\n");
  indent();
  INDENT += 2;

 
  {
    const uint16_t *dataPtr = obj.gainMap().data();
    size_t numElements = obj.gainMap().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("gainMap (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad2x2::CsPad2x2ReadOnlyCfg &obj) {
  printf("CsPad2x2 CsPad2x2ReadOnlyCfg\n");
  indent();
  INDENT += 2;
   indent(); printf("shiftTest: "); print(obj.shiftTest()); printf("\n");
   indent(); printf("version: "); print(obj.version()); printf("\n");

  INDENT -= 2;
}

void print(const CsPad2x2::ElementV1 &obj) {
  printf("CsPad2x2 ElementV1\n");
  indent();
  INDENT += 2;
   indent(); printf("seq_count: "); print(obj.seq_count()); printf("\n");
   indent(); printf("ticks: "); print(obj.ticks()); printf("\n");
   indent(); printf("fiducials: "); print(obj.fiducials()); printf("\n");
   indent(); printf("frame_type: "); print(obj.frame_type()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.sb_temp().data();
    size_t numElements = obj.sb_temp().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("sb_temp (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const int16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const CsPad2x2::ProtectionSystemThreshold &obj) {
  printf("CsPad2x2 ProtectionSystemThreshold\n");
  indent();
  INDENT += 2;
   indent(); printf("adcThreshold: "); print(obj.adcThreshold()); printf("\n");
   indent(); printf("pixelCountThreshold: "); print(obj.pixelCountThreshold()); printf("\n");

  INDENT -= 2;
}

void print(const Encoder::ConfigV1 &obj) {
  printf("Encoder ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("chan_num: "); print(obj.chan_num()); printf("\n");
   indent(); printf("count_mode: "); print(obj.count_mode()); printf("\n");
   indent(); printf("quadrature_mode: "); print(obj.quadrature_mode()); printf("\n");
   indent(); printf("input_num: "); print(obj.input_num()); printf("\n");
   indent(); printf("input_rising: "); print(obj.input_rising()); printf("\n");
   indent(); printf("ticks_per_sec: "); print(obj.ticks_per_sec()); printf("\n");

  INDENT -= 2;
}

void print(const Encoder::ConfigV2 &obj) {
  printf("Encoder ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("chan_mask: "); print(obj.chan_mask()); printf("\n");
   indent(); printf("count_mode: "); print(obj.count_mode()); printf("\n");
   indent(); printf("quadrature_mode: "); print(obj.quadrature_mode()); printf("\n");
   indent(); printf("input_num: "); print(obj.input_num()); printf("\n");
   indent(); printf("input_rising: "); print(obj.input_rising()); printf("\n");
   indent(); printf("ticks_per_sec: "); print(obj.ticks_per_sec()); printf("\n");

  INDENT -= 2;
}

void print(const Encoder::DataV1 &obj) {
  printf("Encoder DataV1\n");
  indent();
  INDENT += 2;
   indent(); printf("timestamp: "); print(obj.timestamp()); printf("\n");
   indent(); printf("encoder_count: "); print(obj.encoder_count()); printf("\n");

  INDENT -= 2;
}

void print(const Encoder::DataV2 &obj) {
  printf("Encoder DataV2\n");
  indent();
  INDENT += 2;
   indent(); printf("timestamp: "); print(obj.timestamp()); printf("\n");

 
  {
    const uint32_t *dataPtr = obj.encoder_count().data();
    size_t numElements = obj.encoder_count().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("encoder_count (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Epix::AsicConfigV1 &obj) {
  printf("Epix AsicConfigV1\n");
  indent();
  INDENT += 2;

  INDENT -= 2;
}

void print(const Epix::ConfigV1 &obj) {
  printf("Epix ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("version: "); print(obj.version()); printf("\n");
   indent(); printf("runTrigDelay: "); print(obj.runTrigDelay()); printf("\n");
   indent(); printf("daqTrigDelay: "); print(obj.daqTrigDelay()); printf("\n");
   indent(); printf("dacSetting: "); print(obj.dacSetting()); printf("\n");
   indent(); printf("acqToAsicR0Delay: "); print(obj.acqToAsicR0Delay()); printf("\n");
   indent(); printf("asicR0ToAsicAcq: "); print(obj.asicR0ToAsicAcq()); printf("\n");
   indent(); printf("asicAcqWidth: "); print(obj.asicAcqWidth()); printf("\n");
   indent(); printf("asicAcqLToPPmatL: "); print(obj.asicAcqLToPPmatL()); printf("\n");
   indent(); printf("asicRoClkHalfT: "); print(obj.asicRoClkHalfT()); printf("\n");
   indent(); printf("adcReadsPerPixel: "); print(obj.adcReadsPerPixel()); printf("\n");
   indent(); printf("adcClkHalfT: "); print(obj.adcClkHalfT()); printf("\n");
   indent(); printf("asicR0Width: "); print(obj.asicR0Width()); printf("\n");
   indent(); printf("adcPipelineDelay: "); print(obj.adcPipelineDelay()); printf("\n");
   indent(); printf("prepulseR0Width: "); print(obj.prepulseR0Width()); printf("\n");
   indent(); printf("prepulseR0Delay: "); print(obj.prepulseR0Delay()); printf("\n");
   indent(); printf("digitalCardId0: "); print(obj.digitalCardId0()); printf("\n");
   indent(); printf("digitalCardId1: "); print(obj.digitalCardId1()); printf("\n");
   indent(); printf("analogCardId0: "); print(obj.analogCardId0()); printf("\n");
   indent(); printf("analogCardId1: "); print(obj.analogCardId1()); printf("\n");
   indent(); printf("lastRowExclusions: "); print(obj.lastRowExclusions()); printf("\n");
   indent(); printf("numberOfAsicsPerRow: "); print(obj.numberOfAsicsPerRow()); printf("\n");
   indent(); printf("numberOfAsicsPerColumn: "); print(obj.numberOfAsicsPerColumn()); printf("\n");
   indent(); printf("numberOfRowsPerAsic: "); print(obj.numberOfRowsPerAsic()); printf("\n");
   indent(); printf("numberOfPixelsPerAsicRow: "); print(obj.numberOfPixelsPerAsicRow()); printf("\n");
   indent(); printf("baseClockFrequency: "); print(obj.baseClockFrequency()); printf("\n");
   indent(); printf("asicMask: "); print(obj.asicMask()); printf("\n");

 
 
  {
    const uint32_t *dataPtr = obj.asicPixelTestArray().data();
    size_t numElements = obj.asicPixelTestArray().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("asicPixelTestArray (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint32_t *dataPtr = obj.asicPixelMaskArray().data();
    size_t numElements = obj.asicPixelMaskArray().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("asicPixelMaskArray (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Epix::ElementV1 &obj) {
  printf("Epix ElementV1\n");
  indent();
  INDENT += 2;
   indent(); printf("frameNumber: "); print(obj.frameNumber()); printf("\n");
   indent(); printf("ticks: "); print(obj.ticks()); printf("\n");
   indent(); printf("fiducials: "); print(obj.fiducials()); printf("\n");
   indent(); printf("lastWord: "); print(obj.lastWord()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.frame().data();
    size_t numElements = obj.frame().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("frame (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint16_t *dataPtr = obj.excludedRows().data();
    size_t numElements = obj.excludedRows().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("excludedRows (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint16_t *dataPtr = obj.temperatures().data();
    size_t numElements = obj.temperatures().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("temperatures (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const EpixSampler::ConfigV1 &obj) {
  printf("EpixSampler ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("version: "); print(obj.version()); printf("\n");
   indent(); printf("runTrigDelay: "); print(obj.runTrigDelay()); printf("\n");
   indent(); printf("daqTrigDelay: "); print(obj.daqTrigDelay()); printf("\n");
   indent(); printf("daqSetting: "); print(obj.daqSetting()); printf("\n");
   indent(); printf("adcClkHalfT: "); print(obj.adcClkHalfT()); printf("\n");
   indent(); printf("adcPipelineDelay: "); print(obj.adcPipelineDelay()); printf("\n");
   indent(); printf("digitalCardId0: "); print(obj.digitalCardId0()); printf("\n");
   indent(); printf("digitalCardId1: "); print(obj.digitalCardId1()); printf("\n");
   indent(); printf("analogCardId0: "); print(obj.analogCardId0()); printf("\n");
   indent(); printf("analogCardId1: "); print(obj.analogCardId1()); printf("\n");
   indent(); printf("numberOfChannels: "); print(obj.numberOfChannels()); printf("\n");
   indent(); printf("samplesPerChannel: "); print(obj.samplesPerChannel()); printf("\n");
   indent(); printf("baseClockFrequency: "); print(obj.baseClockFrequency()); printf("\n");

  INDENT -= 2;
}

void print(const EpixSampler::ElementV1 &obj) {
  printf("EpixSampler ElementV1\n");
  indent();
  INDENT += 2;
   indent(); printf("frameNumber: "); print(obj.frameNumber()); printf("\n");
   indent(); printf("ticks: "); print(obj.ticks()); printf("\n");
   indent(); printf("fiducials: "); print(obj.fiducials()); printf("\n");
   indent(); printf("lastWord: "); print(obj.lastWord()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.frame().data();
    size_t numElements = obj.frame().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("frame (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint16_t *dataPtr = obj.temperatures().data();
    size_t numElements = obj.temperatures().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("temperatures (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const EvrData::ConfigV1 &obj) {
  printf("EvrData ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("npulses: "); print(obj.npulses()); printf("\n");
   indent(); printf("noutputs: "); print(obj.noutputs()); printf("\n");

 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::ConfigV2 &obj) {
  printf("EvrData ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("opcode: "); print(obj.opcode()); printf("\n");
   indent(); printf("npulses: "); print(obj.npulses()); printf("\n");
   indent(); printf("noutputs: "); print(obj.noutputs()); printf("\n");

 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::ConfigV3 &obj) {
  printf("EvrData ConfigV3\n");
  indent();
  INDENT += 2;
   indent(); printf("neventcodes: "); print(obj.neventcodes()); printf("\n");
   indent(); printf("npulses: "); print(obj.npulses()); printf("\n");
   indent(); printf("noutputs: "); print(obj.noutputs()); printf("\n");

 
  {
  }
 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::ConfigV4 &obj) {
  printf("EvrData ConfigV4\n");
  indent();
  INDENT += 2;
   indent(); printf("neventcodes: "); print(obj.neventcodes()); printf("\n");
   indent(); printf("npulses: "); print(obj.npulses()); printf("\n");
   indent(); printf("noutputs: "); print(obj.noutputs()); printf("\n");

 
  {
  }
 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::ConfigV5 &obj) {
  printf("EvrData ConfigV5\n");
  indent();
  INDENT += 2;
   indent(); printf("neventcodes: "); print(obj.neventcodes()); printf("\n");
   indent(); printf("npulses: "); print(obj.npulses()); printf("\n");
   indent(); printf("noutputs: "); print(obj.noutputs()); printf("\n");
   indent(); printf("seq_config: "); print(obj.seq_config()); printf("\n");

 
  {
  }
 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::ConfigV6 &obj) {
  printf("EvrData ConfigV6\n");
  indent();
  INDENT += 2;
   indent(); printf("neventcodes: "); print(obj.neventcodes()); printf("\n");
   indent(); printf("npulses: "); print(obj.npulses()); printf("\n");
   indent(); printf("noutputs: "); print(obj.noutputs()); printf("\n");
   indent(); printf("seq_config: "); print(obj.seq_config()); printf("\n");

 
  {
  }
 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::ConfigV7 &obj) {
  printf("EvrData ConfigV7\n");
  indent();
  INDENT += 2;
   indent(); printf("neventcodes: "); print(obj.neventcodes()); printf("\n");
   indent(); printf("npulses: "); print(obj.npulses()); printf("\n");
   indent(); printf("noutputs: "); print(obj.noutputs()); printf("\n");
   indent(); printf("seq_config: "); print(obj.seq_config()); printf("\n");

 
  {
  }
 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::DataV3 &obj) {
  printf("EvrData DataV3\n");
  indent();
  INDENT += 2;
   indent(); printf("numFifoEvents: "); print(obj.numFifoEvents()); printf("\n");

 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::EventCodeV3 &obj) {
  printf("EvrData EventCodeV3\n");
  indent();
  INDENT += 2;
   indent(); printf("code: "); print(obj.code()); printf("\n");
   indent(); printf("maskTrigger: "); print(obj.maskTrigger()); printf("\n");
   indent(); printf("maskSet: "); print(obj.maskSet()); printf("\n");
   indent(); printf("maskClear: "); print(obj.maskClear()); printf("\n");

  INDENT -= 2;
}

void print(const EvrData::EventCodeV4 &obj) {
  printf("EvrData EventCodeV4\n");
  indent();
  INDENT += 2;
   indent(); printf("code: "); print(obj.code()); printf("\n");
   indent(); printf("reportDelay: "); print(obj.reportDelay()); printf("\n");
   indent(); printf("reportWidth: "); print(obj.reportWidth()); printf("\n");
   indent(); printf("maskTrigger: "); print(obj.maskTrigger()); printf("\n");
   indent(); printf("maskSet: "); print(obj.maskSet()); printf("\n");
   indent(); printf("maskClear: "); print(obj.maskClear()); printf("\n");

  INDENT -= 2;
}

void print(const EvrData::EventCodeV5 &obj) {
  printf("EvrData EventCodeV5\n");
  indent();
  INDENT += 2;
   indent(); printf("code: "); print(obj.code()); printf("\n");
   indent(); printf("reportDelay: "); print(obj.reportDelay()); printf("\n");
   indent(); printf("reportWidth: "); print(obj.reportWidth()); printf("\n");
   indent(); printf("maskTrigger: "); print(obj.maskTrigger()); printf("\n");
   indent(); printf("maskSet: "); print(obj.maskSet()); printf("\n");
   indent(); printf("maskClear: "); print(obj.maskClear()); printf("\n");

 
  printf(" desc: %s\n", obj.desc());
  INDENT -= 2;
}

void print(const EvrData::EventCodeV6 &obj) {
  printf("EvrData EventCodeV6\n");
  indent();
  INDENT += 2;
   indent(); printf("code: "); print(obj.code()); printf("\n");
   indent(); printf("reportDelay: "); print(obj.reportDelay()); printf("\n");
   indent(); printf("reportWidth: "); print(obj.reportWidth()); printf("\n");
   indent(); printf("maskTrigger: "); print(obj.maskTrigger()); printf("\n");
   indent(); printf("maskSet: "); print(obj.maskSet()); printf("\n");
   indent(); printf("maskClear: "); print(obj.maskClear()); printf("\n");
   indent(); printf("readoutGroup: "); print(obj.readoutGroup()); printf("\n");

 
  printf(" desc: %s\n", obj.desc());
  INDENT -= 2;
}

void print(const EvrData::FIFOEvent &obj) {
  printf("EvrData FIFOEvent\n");
  indent();
  INDENT += 2;
   indent(); printf("timestampHigh: "); print(obj.timestampHigh()); printf("\n");
   indent(); printf("timestampLow: "); print(obj.timestampLow()); printf("\n");
   indent(); printf("eventCode: "); print(obj.eventCode()); printf("\n");

  INDENT -= 2;
}

void print(const EvrData::IOChannel &obj) {
  printf("EvrData IOChannel\n");
  indent();
  INDENT += 2;
   indent(); printf("ninfo: "); print(obj.ninfo()); printf("\n");

 
  printf(" name: %s\n", obj.name());
 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::IOConfigV1 &obj) {
  printf("EvrData IOConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("nchannels: "); print(obj.nchannels()); printf("\n");

 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::OutputMap &obj) {
  printf("EvrData OutputMap\n");
  indent();
  INDENT += 2;
   indent(); printf("value: "); print(obj.value()); printf("\n");

  INDENT -= 2;
}

void print(const EvrData::OutputMapV2 &obj) {
  printf("EvrData OutputMapV2\n");
  indent();
  INDENT += 2;
   indent(); printf("value: "); print(obj.value()); printf("\n");

  INDENT -= 2;
}

void print(const EvrData::PulseConfig &obj) {
  printf("EvrData PulseConfig\n");
  indent();
  INDENT += 2;
   indent(); printf("pulse: "); print(obj.pulse()); printf("\n");
   indent(); printf("_input_control_value: "); print(obj._input_control_value()); printf("\n");
   indent(); printf("_output_control_value: "); print(obj._output_control_value()); printf("\n");
   indent(); printf("prescale: "); print(obj.prescale()); printf("\n");
   indent(); printf("delay: "); print(obj.delay()); printf("\n");
   indent(); printf("width: "); print(obj.width()); printf("\n");

  INDENT -= 2;
}

void print(const EvrData::PulseConfigV3 &obj) {
  printf("EvrData PulseConfigV3\n");
  indent();
  INDENT += 2;
   indent(); printf("pulseId: "); print(obj.pulseId()); printf("\n");
   indent(); printf("polarity: "); print(obj.polarity()); printf("\n");
   indent(); printf("prescale: "); print(obj.prescale()); printf("\n");
   indent(); printf("delay: "); print(obj.delay()); printf("\n");
   indent(); printf("width: "); print(obj.width()); printf("\n");

  INDENT -= 2;
}

void print(const EvrData::SequencerConfigV1 &obj) {
  printf("EvrData SequencerConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("length: "); print(obj.length()); printf("\n");
   indent(); printf("cycles: "); print(obj.cycles()); printf("\n");

 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::SequencerEntry &obj) {
  printf("EvrData SequencerEntry\n");
  indent();
  INDENT += 2;

  INDENT -= 2;
}

void print(const EvrData::SrcConfigV1 &obj) {
  printf("EvrData SrcConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("neventcodes: "); print(obj.neventcodes()); printf("\n");
   indent(); printf("npulses: "); print(obj.npulses()); printf("\n");
   indent(); printf("noutputs: "); print(obj.noutputs()); printf("\n");

 
  {
  }
 
  {
  }
 
  {
  }
  INDENT -= 2;
}

void print(const EvrData::SrcEventCode &obj) {
  printf("EvrData SrcEventCode\n");
  indent();
  INDENT += 2;
   indent(); printf("code: "); print(obj.code()); printf("\n");
   indent(); printf("period: "); print(obj.period()); printf("\n");
   indent(); printf("maskTriggerP: "); print(obj.maskTriggerP()); printf("\n");
   indent(); printf("maskTriggerR: "); print(obj.maskTriggerR()); printf("\n");
   indent(); printf("readoutGroup: "); print(obj.readoutGroup()); printf("\n");

 
  printf(" desc: %s\n", obj.desc());
  INDENT -= 2;
}

void print(const FCCD::FccdConfigV1 &obj) {
  printf("FCCD FccdConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("outputMode: "); print(obj.outputMode()); printf("\n");

  INDENT -= 2;
}

void print(const FCCD::FccdConfigV2 &obj) {
  printf("FCCD FccdConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("outputMode: "); print(obj.outputMode()); printf("\n");
   indent(); printf("ccdEnable: "); print(obj.ccdEnable()); printf("\n");
   indent(); printf("focusMode: "); print(obj.focusMode()); printf("\n");
   indent(); printf("exposureTime: "); print(obj.exposureTime()); printf("\n");

 
  {
    const float *dataPtr = obj.dacVoltages().data();
    size_t numElements = obj.dacVoltages().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("dacVoltages (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint16_t *dataPtr = obj.waveforms().data();
    size_t numElements = obj.waveforms().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("waveforms (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Fli::ConfigV1 &obj) {
  printf("Fli ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("width: "); print(obj.width()); printf("\n");
   indent(); printf("height: "); print(obj.height()); printf("\n");
   indent(); printf("orgX: "); print(obj.orgX()); printf("\n");
   indent(); printf("orgY: "); print(obj.orgY()); printf("\n");
   indent(); printf("binX: "); print(obj.binX()); printf("\n");
   indent(); printf("binY: "); print(obj.binY()); printf("\n");
   indent(); printf("exposureTime: "); print(obj.exposureTime()); printf("\n");
   indent(); printf("coolingTemp: "); print(obj.coolingTemp()); printf("\n");
   indent(); printf("gainIndex: "); print(obj.gainIndex()); printf("\n");
   indent(); printf("readoutSpeedIndex: "); print(obj.readoutSpeedIndex()); printf("\n");
   indent(); printf("exposureEventCode: "); print(obj.exposureEventCode()); printf("\n");
   indent(); printf("numDelayShots: "); print(obj.numDelayShots()); printf("\n");

  INDENT -= 2;
}

void print(const Fli::FrameV1 &obj) {
  printf("Fli FrameV1\n");
  indent();
  INDENT += 2;
   indent(); printf("shotIdStart: "); print(obj.shotIdStart()); printf("\n");
   indent(); printf("readoutTime: "); print(obj.readoutTime()); printf("\n");
   indent(); printf("temperature: "); print(obj.temperature()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Gsc16ai::ConfigV1 &obj) {
  printf("Gsc16ai ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("voltageRange: "); print(obj.voltageRange()); printf("\n");
   indent(); printf("firstChan: "); print(obj.firstChan()); printf("\n");
   indent(); printf("lastChan: "); print(obj.lastChan()); printf("\n");
   indent(); printf("inputMode: "); print(obj.inputMode()); printf("\n");
   indent(); printf("triggerMode: "); print(obj.triggerMode()); printf("\n");
   indent(); printf("dataFormat: "); print(obj.dataFormat()); printf("\n");
   indent(); printf("fps: "); print(obj.fps()); printf("\n");
   indent(); printf("autocalibEnable: "); print(obj.autocalibEnable()); printf("\n");
   indent(); printf("timeTagEnable: "); print(obj.timeTagEnable()); printf("\n");

  INDENT -= 2;
}

void print(const Gsc16ai::DataV1 &obj) {
  printf("Gsc16ai DataV1\n");
  indent();
  INDENT += 2;

 
  {
    const uint16_t *dataPtr = obj.timestamp().data();
    size_t numElements = obj.timestamp().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("timestamp (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint16_t *dataPtr = obj.channelValue().data();
    size_t numElements = obj.channelValue().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("channelValue (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Imp::ConfigV1 &obj) {
  printf("Imp ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("range: "); print(obj.range()); printf("\n");
   indent(); printf("calRange: "); print(obj.calRange()); printf("\n");
   indent(); printf("reset: "); print(obj.reset()); printf("\n");
   indent(); printf("biasData: "); print(obj.biasData()); printf("\n");
   indent(); printf("calData: "); print(obj.calData()); printf("\n");
   indent(); printf("biasDacData: "); print(obj.biasDacData()); printf("\n");
   indent(); printf("calStrobe: "); print(obj.calStrobe()); printf("\n");
   indent(); printf("numberOfSamples: "); print(obj.numberOfSamples()); printf("\n");
   indent(); printf("trigDelay: "); print(obj.trigDelay()); printf("\n");
   indent(); printf("adcDelay: "); print(obj.adcDelay()); printf("\n");

  INDENT -= 2;
}

void print(const Imp::ElementV1 &obj) {
  printf("Imp ElementV1\n");
  indent();
  INDENT += 2;
   indent(); printf("frameNumber: "); print(obj.frameNumber()); printf("\n");
   indent(); printf("range: "); print(obj.range()); printf("\n");
   indent(); printf("laneStatus: "); print(obj.laneStatus()); printf("\n");

 
  {
  }
  INDENT -= 2;
}

void print(const Imp::LaneStatus &obj) {
  printf("Imp LaneStatus\n");
  indent();
  INDENT += 2;

  INDENT -= 2;
}

void print(const Imp::Sample &obj) {
  printf("Imp Sample\n");
  indent();
  INDENT += 2;

 
  {
    const uint16_t *dataPtr = obj.channels().data();
    size_t numElements = obj.channels().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("channels (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Ipimb::ConfigV1 &obj) {
  printf("Ipimb ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("triggerCounter: "); print(obj.triggerCounter()); printf("\n");
   indent(); printf("serialID: "); print(obj.serialID()); printf("\n");
   indent(); printf("chargeAmpRange: "); print(obj.chargeAmpRange()); printf("\n");
   indent(); printf("calibrationRange: "); print(obj.calibrationRange()); printf("\n");
   indent(); printf("resetLength: "); print(obj.resetLength()); printf("\n");
   indent(); printf("resetDelay: "); print(obj.resetDelay()); printf("\n");
   indent(); printf("chargeAmpRefVoltage: "); print(obj.chargeAmpRefVoltage()); printf("\n");
   indent(); printf("calibrationVoltage: "); print(obj.calibrationVoltage()); printf("\n");
   indent(); printf("diodeBias: "); print(obj.diodeBias()); printf("\n");
   indent(); printf("status: "); print(obj.status()); printf("\n");
   indent(); printf("errors: "); print(obj.errors()); printf("\n");
   indent(); printf("calStrobeLength: "); print(obj.calStrobeLength()); printf("\n");
   indent(); printf("trigDelay: "); print(obj.trigDelay()); printf("\n");

  INDENT -= 2;
}

void print(const Ipimb::ConfigV2 &obj) {
  printf("Ipimb ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("triggerCounter: "); print(obj.triggerCounter()); printf("\n");
   indent(); printf("serialID: "); print(obj.serialID()); printf("\n");
   indent(); printf("chargeAmpRange: "); print(obj.chargeAmpRange()); printf("\n");
   indent(); printf("calibrationRange: "); print(obj.calibrationRange()); printf("\n");
   indent(); printf("resetLength: "); print(obj.resetLength()); printf("\n");
   indent(); printf("resetDelay: "); print(obj.resetDelay()); printf("\n");
   indent(); printf("chargeAmpRefVoltage: "); print(obj.chargeAmpRefVoltage()); printf("\n");
   indent(); printf("calibrationVoltage: "); print(obj.calibrationVoltage()); printf("\n");
   indent(); printf("diodeBias: "); print(obj.diodeBias()); printf("\n");
   indent(); printf("status: "); print(obj.status()); printf("\n");
   indent(); printf("errors: "); print(obj.errors()); printf("\n");
   indent(); printf("calStrobeLength: "); print(obj.calStrobeLength()); printf("\n");
   indent(); printf("trigDelay: "); print(obj.trigDelay()); printf("\n");
   indent(); printf("trigPsDelay: "); print(obj.trigPsDelay()); printf("\n");
   indent(); printf("adcDelay: "); print(obj.adcDelay()); printf("\n");

  INDENT -= 2;
}

void print(const Ipimb::DataV1 &obj) {
  printf("Ipimb DataV1\n");
  indent();
  INDENT += 2;
   indent(); printf("triggerCounter: "); print(obj.triggerCounter()); printf("\n");
   indent(); printf("config0: "); print(obj.config0()); printf("\n");
   indent(); printf("config1: "); print(obj.config1()); printf("\n");
   indent(); printf("config2: "); print(obj.config2()); printf("\n");
   indent(); printf("channel0: "); print(obj.channel0()); printf("\n");
   indent(); printf("channel1: "); print(obj.channel1()); printf("\n");
   indent(); printf("channel2: "); print(obj.channel2()); printf("\n");
   indent(); printf("channel3: "); print(obj.channel3()); printf("\n");
   indent(); printf("checksum: "); print(obj.checksum()); printf("\n");

  INDENT -= 2;
}

void print(const Ipimb::DataV2 &obj) {
  printf("Ipimb DataV2\n");
  indent();
  INDENT += 2;
   indent(); printf("config0: "); print(obj.config0()); printf("\n");
   indent(); printf("config1: "); print(obj.config1()); printf("\n");
   indent(); printf("config2: "); print(obj.config2()); printf("\n");
   indent(); printf("channel0: "); print(obj.channel0()); printf("\n");
   indent(); printf("channel1: "); print(obj.channel1()); printf("\n");
   indent(); printf("channel2: "); print(obj.channel2()); printf("\n");
   indent(); printf("channel3: "); print(obj.channel3()); printf("\n");
   indent(); printf("channel0ps: "); print(obj.channel0ps()); printf("\n");
   indent(); printf("channel1ps: "); print(obj.channel1ps()); printf("\n");
   indent(); printf("channel2ps: "); print(obj.channel2ps()); printf("\n");
   indent(); printf("channel3ps: "); print(obj.channel3ps()); printf("\n");
   indent(); printf("checksum: "); print(obj.checksum()); printf("\n");

  INDENT -= 2;
}

void print(const L3T::ConfigV1 &obj) {
  printf("L3T ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("module_id_len: "); print(obj.module_id_len()); printf("\n");
   indent(); printf("desc_len: "); print(obj.desc_len()); printf("\n");

 
  printf(" module_id: %s\n", obj.module_id());
 
  printf(" desc: %s\n", obj.desc());
  INDENT -= 2;
}

void print(const L3T::DataV1 &obj) {
  printf("L3T DataV1\n");
  indent();
  INDENT += 2;
   indent(); printf("accept: "); print(obj.accept()); printf("\n");

  INDENT -= 2;
}

void print(const Lusi::DiodeFexConfigV1 &obj) {
  printf("Lusi DiodeFexConfigV1\n");
  indent();
  INDENT += 2;

 
  {
    const float *dataPtr = obj.base().data();
    size_t numElements = obj.base().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("base (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const float *dataPtr = obj.scale().data();
    size_t numElements = obj.scale().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("scale (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Lusi::DiodeFexConfigV2 &obj) {
  printf("Lusi DiodeFexConfigV2\n");
  indent();
  INDENT += 2;

 
  {
    const float *dataPtr = obj.base().data();
    size_t numElements = obj.base().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("base (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const float *dataPtr = obj.scale().data();
    size_t numElements = obj.scale().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("scale (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Lusi::DiodeFexV1 &obj) {
  printf("Lusi DiodeFexV1\n");
  indent();
  INDENT += 2;
   indent(); printf("value: "); print(obj.value()); printf("\n");

  INDENT -= 2;
}

void print(const Lusi::IpmFexConfigV1 &obj) {
  printf("Lusi IpmFexConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("xscale: "); print(obj.xscale()); printf("\n");
   indent(); printf("yscale: "); print(obj.yscale()); printf("\n");

 
  {
  }
  INDENT -= 2;
}

void print(const Lusi::IpmFexConfigV2 &obj) {
  printf("Lusi IpmFexConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("xscale: "); print(obj.xscale()); printf("\n");
   indent(); printf("yscale: "); print(obj.yscale()); printf("\n");

 
  {
  }
  INDENT -= 2;
}

void print(const Lusi::IpmFexV1 &obj) {
  printf("Lusi IpmFexV1\n");
  indent();
  INDENT += 2;
   indent(); printf("sum: "); print(obj.sum()); printf("\n");
   indent(); printf("xpos: "); print(obj.xpos()); printf("\n");
   indent(); printf("ypos: "); print(obj.ypos()); printf("\n");

 
  {
    const float *dataPtr = obj.channel().data();
    size_t numElements = obj.channel().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("channel (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Lusi::PimImageConfigV1 &obj) {
  printf("Lusi PimImageConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("xscale: "); print(obj.xscale()); printf("\n");
   indent(); printf("yscale: "); print(obj.yscale()); printf("\n");

  INDENT -= 2;
}

void print(const OceanOptics::ConfigV1 &obj) {
  printf("OceanOptics ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("exposureTime: "); print(obj.exposureTime()); printf("\n");
   indent(); printf("strayLightConstant: "); print(obj.strayLightConstant()); printf("\n");

 
  {
    const double *dataPtr = obj.waveLenCalib().data();
    size_t numElements = obj.waveLenCalib().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("waveLenCalib (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const double *dataPtr = obj.nonlinCorrect().data();
    size_t numElements = obj.nonlinCorrect().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("nonlinCorrect (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const OceanOptics::DataV1 &obj) {
  printf("OceanOptics DataV1\n");
  indent();
  INDENT += 2;
   indent(); printf("frameCounter: "); print(obj.frameCounter()); printf("\n");
   indent(); printf("numDelayedFrames: "); print(obj.numDelayedFrames()); printf("\n");
   indent(); printf("numDiscardFrames: "); print(obj.numDiscardFrames()); printf("\n");
   indent(); printf("timeFrameStart: "); print(obj.timeFrameStart()); printf("\n");
   indent(); printf("timeFrameFirstData: "); print(obj.timeFrameFirstData()); printf("\n");
   indent(); printf("timeFrameEnd: "); print(obj.timeFrameEnd()); printf("\n");
   indent(); printf("numSpectraInData: "); print(obj.numSpectraInData()); printf("\n");
   indent(); printf("numSpectraInQueue: "); print(obj.numSpectraInQueue()); printf("\n");
   indent(); printf("numSpectraUnused: "); print(obj.numSpectraUnused()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const OceanOptics::timespec64 &obj) {
  printf("OceanOptics timespec64\n");
  indent();
  INDENT += 2;
   indent(); printf("tv_sec: "); print(obj.tv_sec()); printf("\n");
   indent(); printf("tv_nsec: "); print(obj.tv_nsec()); printf("\n");

  INDENT -= 2;
}

void print(const Opal1k::ConfigV1 &obj) {
  printf("Opal1k ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("number_of_defect_pixels: "); print(obj.number_of_defect_pixels()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.output_lookup_table().data();
    size_t numElements = obj.output_lookup_table().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("output_lookup_table (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
  }
  INDENT -= 2;
}

void print(const Orca::ConfigV1 &obj) {
  printf("Orca ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("rows: "); print(obj.rows()); printf("\n");

  INDENT -= 2;
}

void print(const PNCCD::ConfigV1 &obj) {
  printf("PNCCD ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("numLinks: "); print(obj.numLinks()); printf("\n");
   indent(); printf("payloadSizePerLink: "); print(obj.payloadSizePerLink()); printf("\n");

  INDENT -= 2;
}

void print(const PNCCD::ConfigV2 &obj) {
  printf("PNCCD ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("numLinks: "); print(obj.numLinks()); printf("\n");
   indent(); printf("payloadSizePerLink: "); print(obj.payloadSizePerLink()); printf("\n");
   indent(); printf("numChannels: "); print(obj.numChannels()); printf("\n");
   indent(); printf("numRows: "); print(obj.numRows()); printf("\n");
   indent(); printf("numSubmoduleChannels: "); print(obj.numSubmoduleChannels()); printf("\n");
   indent(); printf("numSubmoduleRows: "); print(obj.numSubmoduleRows()); printf("\n");
   indent(); printf("numSubmodules: "); print(obj.numSubmodules()); printf("\n");
   indent(); printf("camexMagic: "); print(obj.camexMagic()); printf("\n");

 
  printf(" info: %s\n", obj.info());
 
  printf(" timingFName: %s\n", obj.timingFName());
  INDENT -= 2;
}

void print(const PNCCD::FrameV1 &obj) {
  printf("PNCCD FrameV1\n");
  indent();
  INDENT += 2;
   indent(); printf("specialWord: "); print(obj.specialWord()); printf("\n");
   indent(); printf("frameNumber: "); print(obj.frameNumber()); printf("\n");
   indent(); printf("timeStampHi: "); print(obj.timeStampHi()); printf("\n");
   indent(); printf("timeStampLo: "); print(obj.timeStampLo()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj._data().data();
    size_t numElements = obj._data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("_data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const PNCCD::FramesV1 &obj) {
  printf("PNCCD FramesV1\n");
  indent();
  INDENT += 2;

 
  INDENT -= 2;
}

void print(const PNCCD::FullFrameV1 &obj) {
  printf("PNCCD FullFrameV1\n");
  indent();
  INDENT += 2;
   indent(); printf("specialWord: "); print(obj.specialWord()); printf("\n");
   indent(); printf("frameNumber: "); print(obj.frameNumber()); printf("\n");
   indent(); printf("timeStampHi: "); print(obj.timeStampHi()); printf("\n");
   indent(); printf("timeStampLo: "); print(obj.timeStampLo()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Pds::ClockTime &obj) {
  printf("Pds ClockTime\n");
  indent();
  INDENT += 2;
   indent(); printf("nanoseconds: "); print(obj.nanoseconds()); printf("\n");
   indent(); printf("seconds: "); print(obj.seconds()); printf("\n");

  INDENT -= 2;
}

void print(const Pds::DetInfo &obj) {
  printf("Pds DetInfo\n");
  indent();
  INDENT += 2;
   indent(); printf("log: "); print(obj.log()); printf("\n");
   indent(); printf("phy: "); print(obj.phy()); printf("\n");

  INDENT -= 2;
}

void print(const Pds::Src &obj) {
  printf("Pds Src\n");
  indent();
  INDENT += 2;
   indent(); printf("log: "); print(obj.log()); printf("\n");
   indent(); printf("phy: "); print(obj.phy()); printf("\n");

  INDENT -= 2;
}

void print(const Princeton::ConfigV1 &obj) {
  printf("Princeton ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("width: "); print(obj.width()); printf("\n");
   indent(); printf("height: "); print(obj.height()); printf("\n");
   indent(); printf("orgX: "); print(obj.orgX()); printf("\n");
   indent(); printf("orgY: "); print(obj.orgY()); printf("\n");
   indent(); printf("binX: "); print(obj.binX()); printf("\n");
   indent(); printf("binY: "); print(obj.binY()); printf("\n");
   indent(); printf("exposureTime: "); print(obj.exposureTime()); printf("\n");
   indent(); printf("coolingTemp: "); print(obj.coolingTemp()); printf("\n");
   indent(); printf("readoutSpeedIndex: "); print(obj.readoutSpeedIndex()); printf("\n");
   indent(); printf("readoutEventCode: "); print(obj.readoutEventCode()); printf("\n");
   indent(); printf("delayMode: "); print(obj.delayMode()); printf("\n");

  INDENT -= 2;
}

void print(const Princeton::ConfigV2 &obj) {
  printf("Princeton ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("width: "); print(obj.width()); printf("\n");
   indent(); printf("height: "); print(obj.height()); printf("\n");
   indent(); printf("orgX: "); print(obj.orgX()); printf("\n");
   indent(); printf("orgY: "); print(obj.orgY()); printf("\n");
   indent(); printf("binX: "); print(obj.binX()); printf("\n");
   indent(); printf("binY: "); print(obj.binY()); printf("\n");
   indent(); printf("exposureTime: "); print(obj.exposureTime()); printf("\n");
   indent(); printf("coolingTemp: "); print(obj.coolingTemp()); printf("\n");
   indent(); printf("gainIndex: "); print(obj.gainIndex()); printf("\n");
   indent(); printf("readoutSpeedIndex: "); print(obj.readoutSpeedIndex()); printf("\n");
   indent(); printf("readoutEventCode: "); print(obj.readoutEventCode()); printf("\n");
   indent(); printf("delayMode: "); print(obj.delayMode()); printf("\n");

  INDENT -= 2;
}

void print(const Princeton::ConfigV3 &obj) {
  printf("Princeton ConfigV3\n");
  indent();
  INDENT += 2;
   indent(); printf("width: "); print(obj.width()); printf("\n");
   indent(); printf("height: "); print(obj.height()); printf("\n");
   indent(); printf("orgX: "); print(obj.orgX()); printf("\n");
   indent(); printf("orgY: "); print(obj.orgY()); printf("\n");
   indent(); printf("binX: "); print(obj.binX()); printf("\n");
   indent(); printf("binY: "); print(obj.binY()); printf("\n");
   indent(); printf("exposureTime: "); print(obj.exposureTime()); printf("\n");
   indent(); printf("coolingTemp: "); print(obj.coolingTemp()); printf("\n");
   indent(); printf("gainIndex: "); print(obj.gainIndex()); printf("\n");
   indent(); printf("readoutSpeedIndex: "); print(obj.readoutSpeedIndex()); printf("\n");
   indent(); printf("exposureEventCode: "); print(obj.exposureEventCode()); printf("\n");
   indent(); printf("numDelayShots: "); print(obj.numDelayShots()); printf("\n");

  INDENT -= 2;
}

void print(const Princeton::ConfigV4 &obj) {
  printf("Princeton ConfigV4\n");
  indent();
  INDENT += 2;
   indent(); printf("width: "); print(obj.width()); printf("\n");
   indent(); printf("height: "); print(obj.height()); printf("\n");
   indent(); printf("orgX: "); print(obj.orgX()); printf("\n");
   indent(); printf("orgY: "); print(obj.orgY()); printf("\n");
   indent(); printf("binX: "); print(obj.binX()); printf("\n");
   indent(); printf("binY: "); print(obj.binY()); printf("\n");
   indent(); printf("maskedHeight: "); print(obj.maskedHeight()); printf("\n");
   indent(); printf("kineticHeight: "); print(obj.kineticHeight()); printf("\n");
   indent(); printf("vsSpeed: "); print(obj.vsSpeed()); printf("\n");
   indent(); printf("exposureTime: "); print(obj.exposureTime()); printf("\n");
   indent(); printf("coolingTemp: "); print(obj.coolingTemp()); printf("\n");
   indent(); printf("gainIndex: "); print(obj.gainIndex()); printf("\n");
   indent(); printf("readoutSpeedIndex: "); print(obj.readoutSpeedIndex()); printf("\n");
   indent(); printf("exposureEventCode: "); print(obj.exposureEventCode()); printf("\n");
   indent(); printf("numDelayShots: "); print(obj.numDelayShots()); printf("\n");

  INDENT -= 2;
}

void print(const Princeton::ConfigV5 &obj) {
  printf("Princeton ConfigV5\n");
  indent();
  INDENT += 2;
   indent(); printf("width: "); print(obj.width()); printf("\n");
   indent(); printf("height: "); print(obj.height()); printf("\n");
   indent(); printf("orgX: "); print(obj.orgX()); printf("\n");
   indent(); printf("orgY: "); print(obj.orgY()); printf("\n");
   indent(); printf("binX: "); print(obj.binX()); printf("\n");
   indent(); printf("binY: "); print(obj.binY()); printf("\n");
   indent(); printf("exposureTime: "); print(obj.exposureTime()); printf("\n");
   indent(); printf("coolingTemp: "); print(obj.coolingTemp()); printf("\n");
   indent(); printf("gainIndex: "); print(obj.gainIndex()); printf("\n");
   indent(); printf("readoutSpeedIndex: "); print(obj.readoutSpeedIndex()); printf("\n");
   indent(); printf("maskedHeight: "); print(obj.maskedHeight()); printf("\n");
   indent(); printf("kineticHeight: "); print(obj.kineticHeight()); printf("\n");
   indent(); printf("vsSpeed: "); print(obj.vsSpeed()); printf("\n");
   indent(); printf("infoReportInterval: "); print(obj.infoReportInterval()); printf("\n");
   indent(); printf("exposureEventCode: "); print(obj.exposureEventCode()); printf("\n");
   indent(); printf("numDelayShots: "); print(obj.numDelayShots()); printf("\n");

  INDENT -= 2;
}

void print(const Princeton::FrameV1 &obj) {
  printf("Princeton FrameV1\n");
  indent();
  INDENT += 2;
   indent(); printf("shotIdStart: "); print(obj.shotIdStart()); printf("\n");
   indent(); printf("readoutTime: "); print(obj.readoutTime()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Princeton::FrameV2 &obj) {
  printf("Princeton FrameV2\n");
  indent();
  INDENT += 2;
   indent(); printf("shotIdStart: "); print(obj.shotIdStart()); printf("\n");
   indent(); printf("readoutTime: "); print(obj.readoutTime()); printf("\n");
   indent(); printf("temperature: "); print(obj.temperature()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Princeton::InfoV1 &obj) {
  printf("Princeton InfoV1\n");
  indent();
  INDENT += 2;
   indent(); printf("temperature: "); print(obj.temperature()); printf("\n");

  INDENT -= 2;
}

void print(const Pulnix::TM6740ConfigV1 &obj) {
  printf("Pulnix TM6740ConfigV1\n");
  indent();
  INDENT += 2;

  INDENT -= 2;
}

void print(const Pulnix::TM6740ConfigV2 &obj) {
  printf("Pulnix TM6740ConfigV2\n");
  indent();
  INDENT += 2;

  INDENT -= 2;
}

void print(const Quartz::ConfigV1 &obj) {
  printf("Quartz ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("number_of_defect_pixels: "); print(obj.number_of_defect_pixels()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.output_lookup_table().data();
    size_t numElements = obj.output_lookup_table().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("output_lookup_table (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
  }
  INDENT -= 2;
}

void print(const Rayonix::ConfigV1 &obj) {
  printf("Rayonix ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("binning_f: "); print(obj.binning_f()); printf("\n");
   indent(); printf("binning_s: "); print(obj.binning_s()); printf("\n");
   indent(); printf("exposure: "); print(obj.exposure()); printf("\n");
   indent(); printf("trigger: "); print(obj.trigger()); printf("\n");
   indent(); printf("rawMode: "); print(obj.rawMode()); printf("\n");
   indent(); printf("darkFlag: "); print(obj.darkFlag()); printf("\n");
   indent(); printf("readoutMode: "); print(obj.readoutMode()); printf("\n");

 
  printf(" deviceID: %s\n", obj.deviceID());
  INDENT -= 2;
}

void print(const Rayonix::ConfigV2 &obj) {
  printf("Rayonix ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("binning_f: "); print(obj.binning_f()); printf("\n");
   indent(); printf("binning_s: "); print(obj.binning_s()); printf("\n");
   indent(); printf("testPattern: "); print(obj.testPattern()); printf("\n");
   indent(); printf("exposure: "); print(obj.exposure()); printf("\n");
   indent(); printf("trigger: "); print(obj.trigger()); printf("\n");
   indent(); printf("rawMode: "); print(obj.rawMode()); printf("\n");
   indent(); printf("darkFlag: "); print(obj.darkFlag()); printf("\n");
   indent(); printf("readoutMode: "); print(obj.readoutMode()); printf("\n");

 
  printf(" deviceID: %s\n", obj.deviceID());
  INDENT -= 2;
}

void print(const Timepix::ConfigV1 &obj) {
  printf("Timepix ConfigV1\n");
  indent();
  INDENT += 2;
   indent(); printf("readoutSpeed: "); print(obj.readoutSpeed()); printf("\n");
   indent(); printf("triggerMode: "); print(obj.triggerMode()); printf("\n");
   indent(); printf("shutterTimeout: "); print(obj.shutterTimeout()); printf("\n");
   indent(); printf("dac0Ikrum: "); print(obj.dac0Ikrum()); printf("\n");
   indent(); printf("dac0Disc: "); print(obj.dac0Disc()); printf("\n");
   indent(); printf("dac0Preamp: "); print(obj.dac0Preamp()); printf("\n");
   indent(); printf("dac0BufAnalogA: "); print(obj.dac0BufAnalogA()); printf("\n");
   indent(); printf("dac0BufAnalogB: "); print(obj.dac0BufAnalogB()); printf("\n");
   indent(); printf("dac0Hist: "); print(obj.dac0Hist()); printf("\n");
   indent(); printf("dac0ThlFine: "); print(obj.dac0ThlFine()); printf("\n");
   indent(); printf("dac0ThlCourse: "); print(obj.dac0ThlCourse()); printf("\n");
   indent(); printf("dac0Vcas: "); print(obj.dac0Vcas()); printf("\n");
   indent(); printf("dac0Fbk: "); print(obj.dac0Fbk()); printf("\n");
   indent(); printf("dac0Gnd: "); print(obj.dac0Gnd()); printf("\n");
   indent(); printf("dac0Ths: "); print(obj.dac0Ths()); printf("\n");
   indent(); printf("dac0BiasLvds: "); print(obj.dac0BiasLvds()); printf("\n");
   indent(); printf("dac0RefLvds: "); print(obj.dac0RefLvds()); printf("\n");
   indent(); printf("dac1Ikrum: "); print(obj.dac1Ikrum()); printf("\n");
   indent(); printf("dac1Disc: "); print(obj.dac1Disc()); printf("\n");
   indent(); printf("dac1Preamp: "); print(obj.dac1Preamp()); printf("\n");
   indent(); printf("dac1BufAnalogA: "); print(obj.dac1BufAnalogA()); printf("\n");
   indent(); printf("dac1BufAnalogB: "); print(obj.dac1BufAnalogB()); printf("\n");
   indent(); printf("dac1Hist: "); print(obj.dac1Hist()); printf("\n");
   indent(); printf("dac1ThlFine: "); print(obj.dac1ThlFine()); printf("\n");
   indent(); printf("dac1ThlCourse: "); print(obj.dac1ThlCourse()); printf("\n");
   indent(); printf("dac1Vcas: "); print(obj.dac1Vcas()); printf("\n");
   indent(); printf("dac1Fbk: "); print(obj.dac1Fbk()); printf("\n");
   indent(); printf("dac1Gnd: "); print(obj.dac1Gnd()); printf("\n");
   indent(); printf("dac1Ths: "); print(obj.dac1Ths()); printf("\n");
   indent(); printf("dac1BiasLvds: "); print(obj.dac1BiasLvds()); printf("\n");
   indent(); printf("dac1RefLvds: "); print(obj.dac1RefLvds()); printf("\n");
   indent(); printf("dac2Ikrum: "); print(obj.dac2Ikrum()); printf("\n");
   indent(); printf("dac2Disc: "); print(obj.dac2Disc()); printf("\n");
   indent(); printf("dac2Preamp: "); print(obj.dac2Preamp()); printf("\n");
   indent(); printf("dac2BufAnalogA: "); print(obj.dac2BufAnalogA()); printf("\n");
   indent(); printf("dac2BufAnalogB: "); print(obj.dac2BufAnalogB()); printf("\n");
   indent(); printf("dac2Hist: "); print(obj.dac2Hist()); printf("\n");
   indent(); printf("dac2ThlFine: "); print(obj.dac2ThlFine()); printf("\n");
   indent(); printf("dac2ThlCourse: "); print(obj.dac2ThlCourse()); printf("\n");
   indent(); printf("dac2Vcas: "); print(obj.dac2Vcas()); printf("\n");
   indent(); printf("dac2Fbk: "); print(obj.dac2Fbk()); printf("\n");
   indent(); printf("dac2Gnd: "); print(obj.dac2Gnd()); printf("\n");
   indent(); printf("dac2Ths: "); print(obj.dac2Ths()); printf("\n");
   indent(); printf("dac2BiasLvds: "); print(obj.dac2BiasLvds()); printf("\n");
   indent(); printf("dac2RefLvds: "); print(obj.dac2RefLvds()); printf("\n");
   indent(); printf("dac3Ikrum: "); print(obj.dac3Ikrum()); printf("\n");
   indent(); printf("dac3Disc: "); print(obj.dac3Disc()); printf("\n");
   indent(); printf("dac3Preamp: "); print(obj.dac3Preamp()); printf("\n");
   indent(); printf("dac3BufAnalogA: "); print(obj.dac3BufAnalogA()); printf("\n");
   indent(); printf("dac3BufAnalogB: "); print(obj.dac3BufAnalogB()); printf("\n");
   indent(); printf("dac3Hist: "); print(obj.dac3Hist()); printf("\n");
   indent(); printf("dac3ThlFine: "); print(obj.dac3ThlFine()); printf("\n");
   indent(); printf("dac3ThlCourse: "); print(obj.dac3ThlCourse()); printf("\n");
   indent(); printf("dac3Vcas: "); print(obj.dac3Vcas()); printf("\n");
   indent(); printf("dac3Fbk: "); print(obj.dac3Fbk()); printf("\n");
   indent(); printf("dac3Gnd: "); print(obj.dac3Gnd()); printf("\n");
   indent(); printf("dac3Ths: "); print(obj.dac3Ths()); printf("\n");
   indent(); printf("dac3BiasLvds: "); print(obj.dac3BiasLvds()); printf("\n");
   indent(); printf("dac3RefLvds: "); print(obj.dac3RefLvds()); printf("\n");

  INDENT -= 2;
}

void print(const Timepix::ConfigV2 &obj) {
  printf("Timepix ConfigV2\n");
  indent();
  INDENT += 2;
   indent(); printf("readoutSpeed: "); print(obj.readoutSpeed()); printf("\n");
   indent(); printf("triggerMode: "); print(obj.triggerMode()); printf("\n");
   indent(); printf("timepixSpeed: "); print(obj.timepixSpeed()); printf("\n");
   indent(); printf("dac0Ikrum: "); print(obj.dac0Ikrum()); printf("\n");
   indent(); printf("dac0Disc: "); print(obj.dac0Disc()); printf("\n");
   indent(); printf("dac0Preamp: "); print(obj.dac0Preamp()); printf("\n");
   indent(); printf("dac0BufAnalogA: "); print(obj.dac0BufAnalogA()); printf("\n");
   indent(); printf("dac0BufAnalogB: "); print(obj.dac0BufAnalogB()); printf("\n");
   indent(); printf("dac0Hist: "); print(obj.dac0Hist()); printf("\n");
   indent(); printf("dac0ThlFine: "); print(obj.dac0ThlFine()); printf("\n");
   indent(); printf("dac0ThlCourse: "); print(obj.dac0ThlCourse()); printf("\n");
   indent(); printf("dac0Vcas: "); print(obj.dac0Vcas()); printf("\n");
   indent(); printf("dac0Fbk: "); print(obj.dac0Fbk()); printf("\n");
   indent(); printf("dac0Gnd: "); print(obj.dac0Gnd()); printf("\n");
   indent(); printf("dac0Ths: "); print(obj.dac0Ths()); printf("\n");
   indent(); printf("dac0BiasLvds: "); print(obj.dac0BiasLvds()); printf("\n");
   indent(); printf("dac0RefLvds: "); print(obj.dac0RefLvds()); printf("\n");
   indent(); printf("dac1Ikrum: "); print(obj.dac1Ikrum()); printf("\n");
   indent(); printf("dac1Disc: "); print(obj.dac1Disc()); printf("\n");
   indent(); printf("dac1Preamp: "); print(obj.dac1Preamp()); printf("\n");
   indent(); printf("dac1BufAnalogA: "); print(obj.dac1BufAnalogA()); printf("\n");
   indent(); printf("dac1BufAnalogB: "); print(obj.dac1BufAnalogB()); printf("\n");
   indent(); printf("dac1Hist: "); print(obj.dac1Hist()); printf("\n");
   indent(); printf("dac1ThlFine: "); print(obj.dac1ThlFine()); printf("\n");
   indent(); printf("dac1ThlCourse: "); print(obj.dac1ThlCourse()); printf("\n");
   indent(); printf("dac1Vcas: "); print(obj.dac1Vcas()); printf("\n");
   indent(); printf("dac1Fbk: "); print(obj.dac1Fbk()); printf("\n");
   indent(); printf("dac1Gnd: "); print(obj.dac1Gnd()); printf("\n");
   indent(); printf("dac1Ths: "); print(obj.dac1Ths()); printf("\n");
   indent(); printf("dac1BiasLvds: "); print(obj.dac1BiasLvds()); printf("\n");
   indent(); printf("dac1RefLvds: "); print(obj.dac1RefLvds()); printf("\n");
   indent(); printf("dac2Ikrum: "); print(obj.dac2Ikrum()); printf("\n");
   indent(); printf("dac2Disc: "); print(obj.dac2Disc()); printf("\n");
   indent(); printf("dac2Preamp: "); print(obj.dac2Preamp()); printf("\n");
   indent(); printf("dac2BufAnalogA: "); print(obj.dac2BufAnalogA()); printf("\n");
   indent(); printf("dac2BufAnalogB: "); print(obj.dac2BufAnalogB()); printf("\n");
   indent(); printf("dac2Hist: "); print(obj.dac2Hist()); printf("\n");
   indent(); printf("dac2ThlFine: "); print(obj.dac2ThlFine()); printf("\n");
   indent(); printf("dac2ThlCourse: "); print(obj.dac2ThlCourse()); printf("\n");
   indent(); printf("dac2Vcas: "); print(obj.dac2Vcas()); printf("\n");
   indent(); printf("dac2Fbk: "); print(obj.dac2Fbk()); printf("\n");
   indent(); printf("dac2Gnd: "); print(obj.dac2Gnd()); printf("\n");
   indent(); printf("dac2Ths: "); print(obj.dac2Ths()); printf("\n");
   indent(); printf("dac2BiasLvds: "); print(obj.dac2BiasLvds()); printf("\n");
   indent(); printf("dac2RefLvds: "); print(obj.dac2RefLvds()); printf("\n");
   indent(); printf("dac3Ikrum: "); print(obj.dac3Ikrum()); printf("\n");
   indent(); printf("dac3Disc: "); print(obj.dac3Disc()); printf("\n");
   indent(); printf("dac3Preamp: "); print(obj.dac3Preamp()); printf("\n");
   indent(); printf("dac3BufAnalogA: "); print(obj.dac3BufAnalogA()); printf("\n");
   indent(); printf("dac3BufAnalogB: "); print(obj.dac3BufAnalogB()); printf("\n");
   indent(); printf("dac3Hist: "); print(obj.dac3Hist()); printf("\n");
   indent(); printf("dac3ThlFine: "); print(obj.dac3ThlFine()); printf("\n");
   indent(); printf("dac3ThlCourse: "); print(obj.dac3ThlCourse()); printf("\n");
   indent(); printf("dac3Vcas: "); print(obj.dac3Vcas()); printf("\n");
   indent(); printf("dac3Fbk: "); print(obj.dac3Fbk()); printf("\n");
   indent(); printf("dac3Gnd: "); print(obj.dac3Gnd()); printf("\n");
   indent(); printf("dac3Ths: "); print(obj.dac3Ths()); printf("\n");
   indent(); printf("dac3BiasLvds: "); print(obj.dac3BiasLvds()); printf("\n");
   indent(); printf("dac3RefLvds: "); print(obj.dac3RefLvds()); printf("\n");
   indent(); printf("driverVersion: "); print(obj.driverVersion()); printf("\n");
   indent(); printf("firmwareVersion: "); print(obj.firmwareVersion()); printf("\n");
   indent(); printf("pixelThreshSize: "); print(obj.pixelThreshSize()); printf("\n");
   indent(); printf("chip0ID: "); print(obj.chip0ID()); printf("\n");
   indent(); printf("chip1ID: "); print(obj.chip1ID()); printf("\n");
   indent(); printf("chip2ID: "); print(obj.chip2ID()); printf("\n");
   indent(); printf("chip3ID: "); print(obj.chip3ID()); printf("\n");

 
  {
    const uint8_t *dataPtr = obj.pixelThresh().data();
    size_t numElements = obj.pixelThresh().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("pixelThresh (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  printf(" chip0Name: %s\n", obj.chip0Name());
 
  printf(" chip1Name: %s\n", obj.chip1Name());
 
  printf(" chip2Name: %s\n", obj.chip2Name());
 
  printf(" chip3Name: %s\n", obj.chip3Name());
  INDENT -= 2;
}

void print(const Timepix::ConfigV3 &obj) {
  printf("Timepix ConfigV3\n");
  indent();
  INDENT += 2;
   indent(); printf("readoutSpeed: "); print(obj.readoutSpeed()); printf("\n");
   indent(); printf("timepixMode: "); print(obj.timepixMode()); printf("\n");
   indent(); printf("timepixSpeed: "); print(obj.timepixSpeed()); printf("\n");
   indent(); printf("dac0Ikrum: "); print(obj.dac0Ikrum()); printf("\n");
   indent(); printf("dac0Disc: "); print(obj.dac0Disc()); printf("\n");
   indent(); printf("dac0Preamp: "); print(obj.dac0Preamp()); printf("\n");
   indent(); printf("dac0BufAnalogA: "); print(obj.dac0BufAnalogA()); printf("\n");
   indent(); printf("dac0BufAnalogB: "); print(obj.dac0BufAnalogB()); printf("\n");
   indent(); printf("dac0Hist: "); print(obj.dac0Hist()); printf("\n");
   indent(); printf("dac0ThlFine: "); print(obj.dac0ThlFine()); printf("\n");
   indent(); printf("dac0ThlCourse: "); print(obj.dac0ThlCourse()); printf("\n");
   indent(); printf("dac0Vcas: "); print(obj.dac0Vcas()); printf("\n");
   indent(); printf("dac0Fbk: "); print(obj.dac0Fbk()); printf("\n");
   indent(); printf("dac0Gnd: "); print(obj.dac0Gnd()); printf("\n");
   indent(); printf("dac0Ths: "); print(obj.dac0Ths()); printf("\n");
   indent(); printf("dac0BiasLvds: "); print(obj.dac0BiasLvds()); printf("\n");
   indent(); printf("dac0RefLvds: "); print(obj.dac0RefLvds()); printf("\n");
   indent(); printf("dac1Ikrum: "); print(obj.dac1Ikrum()); printf("\n");
   indent(); printf("dac1Disc: "); print(obj.dac1Disc()); printf("\n");
   indent(); printf("dac1Preamp: "); print(obj.dac1Preamp()); printf("\n");
   indent(); printf("dac1BufAnalogA: "); print(obj.dac1BufAnalogA()); printf("\n");
   indent(); printf("dac1BufAnalogB: "); print(obj.dac1BufAnalogB()); printf("\n");
   indent(); printf("dac1Hist: "); print(obj.dac1Hist()); printf("\n");
   indent(); printf("dac1ThlFine: "); print(obj.dac1ThlFine()); printf("\n");
   indent(); printf("dac1ThlCourse: "); print(obj.dac1ThlCourse()); printf("\n");
   indent(); printf("dac1Vcas: "); print(obj.dac1Vcas()); printf("\n");
   indent(); printf("dac1Fbk: "); print(obj.dac1Fbk()); printf("\n");
   indent(); printf("dac1Gnd: "); print(obj.dac1Gnd()); printf("\n");
   indent(); printf("dac1Ths: "); print(obj.dac1Ths()); printf("\n");
   indent(); printf("dac1BiasLvds: "); print(obj.dac1BiasLvds()); printf("\n");
   indent(); printf("dac1RefLvds: "); print(obj.dac1RefLvds()); printf("\n");
   indent(); printf("dac2Ikrum: "); print(obj.dac2Ikrum()); printf("\n");
   indent(); printf("dac2Disc: "); print(obj.dac2Disc()); printf("\n");
   indent(); printf("dac2Preamp: "); print(obj.dac2Preamp()); printf("\n");
   indent(); printf("dac2BufAnalogA: "); print(obj.dac2BufAnalogA()); printf("\n");
   indent(); printf("dac2BufAnalogB: "); print(obj.dac2BufAnalogB()); printf("\n");
   indent(); printf("dac2Hist: "); print(obj.dac2Hist()); printf("\n");
   indent(); printf("dac2ThlFine: "); print(obj.dac2ThlFine()); printf("\n");
   indent(); printf("dac2ThlCourse: "); print(obj.dac2ThlCourse()); printf("\n");
   indent(); printf("dac2Vcas: "); print(obj.dac2Vcas()); printf("\n");
   indent(); printf("dac2Fbk: "); print(obj.dac2Fbk()); printf("\n");
   indent(); printf("dac2Gnd: "); print(obj.dac2Gnd()); printf("\n");
   indent(); printf("dac2Ths: "); print(obj.dac2Ths()); printf("\n");
   indent(); printf("dac2BiasLvds: "); print(obj.dac2BiasLvds()); printf("\n");
   indent(); printf("dac2RefLvds: "); print(obj.dac2RefLvds()); printf("\n");
   indent(); printf("dac3Ikrum: "); print(obj.dac3Ikrum()); printf("\n");
   indent(); printf("dac3Disc: "); print(obj.dac3Disc()); printf("\n");
   indent(); printf("dac3Preamp: "); print(obj.dac3Preamp()); printf("\n");
   indent(); printf("dac3BufAnalogA: "); print(obj.dac3BufAnalogA()); printf("\n");
   indent(); printf("dac3BufAnalogB: "); print(obj.dac3BufAnalogB()); printf("\n");
   indent(); printf("dac3Hist: "); print(obj.dac3Hist()); printf("\n");
   indent(); printf("dac3ThlFine: "); print(obj.dac3ThlFine()); printf("\n");
   indent(); printf("dac3ThlCourse: "); print(obj.dac3ThlCourse()); printf("\n");
   indent(); printf("dac3Vcas: "); print(obj.dac3Vcas()); printf("\n");
   indent(); printf("dac3Fbk: "); print(obj.dac3Fbk()); printf("\n");
   indent(); printf("dac3Gnd: "); print(obj.dac3Gnd()); printf("\n");
   indent(); printf("dac3Ths: "); print(obj.dac3Ths()); printf("\n");
   indent(); printf("dac3BiasLvds: "); print(obj.dac3BiasLvds()); printf("\n");
   indent(); printf("dac3RefLvds: "); print(obj.dac3RefLvds()); printf("\n");
   indent(); printf("dacBias: "); print(obj.dacBias()); printf("\n");
   indent(); printf("flags: "); print(obj.flags()); printf("\n");
   indent(); printf("driverVersion: "); print(obj.driverVersion()); printf("\n");
   indent(); printf("firmwareVersion: "); print(obj.firmwareVersion()); printf("\n");
   indent(); printf("pixelThreshSize: "); print(obj.pixelThreshSize()); printf("\n");
   indent(); printf("chip0ID: "); print(obj.chip0ID()); printf("\n");
   indent(); printf("chip1ID: "); print(obj.chip1ID()); printf("\n");
   indent(); printf("chip2ID: "); print(obj.chip2ID()); printf("\n");
   indent(); printf("chip3ID: "); print(obj.chip3ID()); printf("\n");

 
  {
    const uint8_t *dataPtr = obj.pixelThresh().data();
    size_t numElements = obj.pixelThresh().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("pixelThresh (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  printf(" chip0Name: %s\n", obj.chip0Name());
 
  printf(" chip1Name: %s\n", obj.chip1Name());
 
  printf(" chip2Name: %s\n", obj.chip2Name());
 
  printf(" chip3Name: %s\n", obj.chip3Name());
  INDENT -= 2;
}

void print(const Timepix::DataV1 &obj) {
  printf("Timepix DataV1\n");
  indent();
  INDENT += 2;
   indent(); printf("timestamp: "); print(obj.timestamp()); printf("\n");
   indent(); printf("frameCounter: "); print(obj.frameCounter()); printf("\n");
   indent(); printf("lostRows: "); print(obj.lostRows()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const Timepix::DataV2 &obj) {
  printf("Timepix DataV2\n");
  indent();
  INDENT += 2;
   indent(); printf("width: "); print(obj.width()); printf("\n");
   indent(); printf("height: "); print(obj.height()); printf("\n");
   indent(); printf("timestamp: "); print(obj.timestamp()); printf("\n");
   indent(); printf("frameCounter: "); print(obj.frameCounter()); printf("\n");
   indent(); printf("lostRows: "); print(obj.lostRows()); printf("\n");

 
  {
    const uint16_t *dataPtr = obj.data().data();
    size_t numElements = obj.data().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("data (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const UsdUsb::ConfigV1 &obj) {
  printf("UsdUsb ConfigV1\n");
  indent();
  INDENT += 2;

 
  {
    const uint32_t *dataPtr = obj.counting_mode().data();
    size_t numElements = obj.counting_mode().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("counting_mode (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint32_t *dataPtr = obj.quadrature_mode().data();
    size_t numElements = obj.quadrature_mode().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("quadrature_mode (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}

void print(const UsdUsb::DataV1 &obj) {
  printf("UsdUsb DataV1\n");
  indent();
  INDENT += 2;
   indent(); printf("digital_in: "); print(obj.digital_in()); printf("\n");
   indent(); printf("timestamp: "); print(obj.timestamp()); printf("\n");

 
  {
    const uint8_t *dataPtr = obj.status().data();
    size_t numElements = obj.status().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("status (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
 
  {
    const uint16_t *dataPtr = obj.analog_in().data();
    size_t numElements = obj.analog_in().size();
    vector<int> inds = arrayInds(numElements);
    indent(); printf("analog_in (ndarray with %lu elements): ", numElements); 
    for (unsigned i=0; i < inds.size(); ++i) {
      printf(" [%d]=",inds[i]); 
      print(dataPtr[inds[i]]);
    }
    printf("\n");
  }
  INDENT -= 2;
}


  /*
void print(const Epics::EpicsPvCtrlString &obj) {
  printf("Epics EpicsPvCtrlString\n");
  INDENT += 2;
  indent(); printf("dbr: "); print(obj.dbr());
  for (int i = 0; i < obj.numElements(); ++i) {
    indent(); printf("string(%d): %s\n",i,obj.value(i));
  }
  INDENT -= 2;
}
  */
} // PrintPsddl namespace

namespace PsanaTools {
void getAndDumpPsddlObject(PSEvt::Event &evt, PSEnv::Env &env, PSEvt::EventKey &eventKey, bool inEvt) { 
  const std::type_info & keyCppType = *eventKey.typeinfo();
  const Pds::Src &src = eventKey.src();
  const string &key = eventKey.key();

  if (keyCppType == typeid(Acqiris::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Acqiris::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Acqiris::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Acqiris::DataDescV1)) {
    if (inEvt) {
      boost::shared_ptr<Acqiris::DataDescV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Acqiris::DataDescV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Acqiris::TdcConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Acqiris::TdcConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Acqiris::TdcConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Acqiris::TdcDataV1)) {
    if (inEvt) {
      boost::shared_ptr<Acqiris::TdcDataV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Acqiris::TdcDataV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Alias::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Alias::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Alias::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Andor::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Andor::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Andor::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Andor::FrameV1)) {
    if (inEvt) {
      boost::shared_ptr<Andor::FrameV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Andor::FrameV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataAcqADCV1)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataAcqADCV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataAcqADCV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataEBeamV0)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataEBeamV0> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataEBeamV0> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataEBeamV1)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataEBeamV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataEBeamV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataEBeamV2)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataEBeamV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataEBeamV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataEBeamV3)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataEBeamV3> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataEBeamV3> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataEBeamV4)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataEBeamV4> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataEBeamV4> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataFEEGasDetEnergy)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataFEEGasDetEnergy> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataFEEGasDetEnergy> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataGMDV0)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataGMDV0> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataGMDV0> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataGMDV1)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataGMDV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataGMDV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataIpimbV0)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataIpimbV0> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataIpimbV0> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataIpimbV1)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataIpimbV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataIpimbV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataPhaseCavity)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataPhaseCavity> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataPhaseCavity> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataPimV1)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataPimV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataPimV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Bld::BldDataSpectrometerV0)) {
    if (inEvt) {
      boost::shared_ptr<Bld::BldDataSpectrometerV0> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Bld::BldDataSpectrometerV0> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Camera::FrameFccdConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Camera::FrameFccdConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Camera::FrameFccdConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Camera::FrameFexConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Camera::FrameFexConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Camera::FrameFexConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Camera::FrameV1)) {
    if (inEvt) {
      boost::shared_ptr<Camera::FrameV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Camera::FrameV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Camera::TwoDGaussianV1)) {
    if (inEvt) {
      boost::shared_ptr<Camera::TwoDGaussianV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Camera::TwoDGaussianV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(ControlData::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<ControlData::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<ControlData::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(ControlData::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<ControlData::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<ControlData::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(ControlData::ConfigV3)) {
    if (inEvt) {
      boost::shared_ptr<ControlData::ConfigV3> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<ControlData::ConfigV3> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<CsPad::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<CsPad::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad::ConfigV3)) {
    if (inEvt) {
      boost::shared_ptr<CsPad::ConfigV3> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad::ConfigV3> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad::ConfigV4)) {
    if (inEvt) {
      boost::shared_ptr<CsPad::ConfigV4> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad::ConfigV4> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad::ConfigV5)) {
    if (inEvt) {
      boost::shared_ptr<CsPad::ConfigV5> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad::ConfigV5> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad::DataV1)) {
    if (inEvt) {
      boost::shared_ptr<CsPad::DataV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad::DataV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad::DataV2)) {
    if (inEvt) {
      boost::shared_ptr<CsPad::DataV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad::DataV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad2x2::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<CsPad2x2::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad2x2::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad2x2::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<CsPad2x2::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad2x2::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(CsPad2x2::ElementV1)) {
    if (inEvt) {
      boost::shared_ptr<CsPad2x2::ElementV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<CsPad2x2::ElementV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Encoder::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Encoder::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Encoder::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Encoder::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<Encoder::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Encoder::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Encoder::DataV1)) {
    if (inEvt) {
      boost::shared_ptr<Encoder::DataV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Encoder::DataV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Encoder::DataV2)) {
    if (inEvt) {
      boost::shared_ptr<Encoder::DataV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Encoder::DataV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Epix::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Epix::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Epix::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Epix::ElementV1)) {
    if (inEvt) {
      boost::shared_ptr<Epix::ElementV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Epix::ElementV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EpixSampler::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<EpixSampler::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EpixSampler::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EpixSampler::ElementV1)) {
    if (inEvt) {
      boost::shared_ptr<EpixSampler::ElementV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EpixSampler::ElementV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::ConfigV3)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::ConfigV3> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::ConfigV3> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::ConfigV4)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::ConfigV4> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::ConfigV4> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::ConfigV5)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::ConfigV5> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::ConfigV5> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::ConfigV6)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::ConfigV6> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::ConfigV6> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::ConfigV7)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::ConfigV7> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::ConfigV7> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::DataV3)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::DataV3> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::DataV3> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::IOConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::IOConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::IOConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(EvrData::SrcConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<EvrData::SrcConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<EvrData::SrcConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(FCCD::FccdConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<FCCD::FccdConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<FCCD::FccdConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(FCCD::FccdConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<FCCD::FccdConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<FCCD::FccdConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Fli::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Fli::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Fli::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Fli::FrameV1)) {
    if (inEvt) {
      boost::shared_ptr<Fli::FrameV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Fli::FrameV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Gsc16ai::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Gsc16ai::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Gsc16ai::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Gsc16ai::DataV1)) {
    if (inEvt) {
      boost::shared_ptr<Gsc16ai::DataV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Gsc16ai::DataV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Imp::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Imp::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Imp::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Imp::ElementV1)) {
    if (inEvt) {
      boost::shared_ptr<Imp::ElementV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Imp::ElementV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Ipimb::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Ipimb::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Ipimb::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Ipimb::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<Ipimb::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Ipimb::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Ipimb::DataV1)) {
    if (inEvt) {
      boost::shared_ptr<Ipimb::DataV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Ipimb::DataV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Ipimb::DataV2)) {
    if (inEvt) {
      boost::shared_ptr<Ipimb::DataV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Ipimb::DataV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(L3T::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<L3T::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<L3T::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(L3T::DataV1)) {
    if (inEvt) {
      boost::shared_ptr<L3T::DataV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<L3T::DataV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Lusi::DiodeFexConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Lusi::DiodeFexConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Lusi::DiodeFexConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Lusi::DiodeFexConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<Lusi::DiodeFexConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Lusi::DiodeFexConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Lusi::DiodeFexV1)) {
    if (inEvt) {
      boost::shared_ptr<Lusi::DiodeFexV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Lusi::DiodeFexV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Lusi::IpmFexConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Lusi::IpmFexConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Lusi::IpmFexConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Lusi::IpmFexConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<Lusi::IpmFexConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Lusi::IpmFexConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Lusi::IpmFexV1)) {
    if (inEvt) {
      boost::shared_ptr<Lusi::IpmFexV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Lusi::IpmFexV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Lusi::PimImageConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Lusi::PimImageConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Lusi::PimImageConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(OceanOptics::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<OceanOptics::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<OceanOptics::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(OceanOptics::DataV1)) {
    if (inEvt) {
      boost::shared_ptr<OceanOptics::DataV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<OceanOptics::DataV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Opal1k::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Opal1k::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Opal1k::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Orca::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Orca::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Orca::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(PNCCD::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<PNCCD::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<PNCCD::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(PNCCD::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<PNCCD::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<PNCCD::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(PNCCD::FramesV1)) {
    if (inEvt) {
      boost::shared_ptr<PNCCD::FramesV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<PNCCD::FramesV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(PNCCD::FullFrameV1)) {
    if (inEvt) {
      boost::shared_ptr<PNCCD::FullFrameV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<PNCCD::FullFrameV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Princeton::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Princeton::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Princeton::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Princeton::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<Princeton::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Princeton::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Princeton::ConfigV3)) {
    if (inEvt) {
      boost::shared_ptr<Princeton::ConfigV3> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Princeton::ConfigV3> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Princeton::ConfigV4)) {
    if (inEvt) {
      boost::shared_ptr<Princeton::ConfigV4> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Princeton::ConfigV4> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Princeton::ConfigV5)) {
    if (inEvt) {
      boost::shared_ptr<Princeton::ConfigV5> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Princeton::ConfigV5> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Princeton::FrameV1)) {
    if (inEvt) {
      boost::shared_ptr<Princeton::FrameV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Princeton::FrameV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Princeton::FrameV2)) {
    if (inEvt) {
      boost::shared_ptr<Princeton::FrameV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Princeton::FrameV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Princeton::InfoV1)) {
    if (inEvt) {
      boost::shared_ptr<Princeton::InfoV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Princeton::InfoV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Pulnix::TM6740ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Pulnix::TM6740ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Pulnix::TM6740ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Pulnix::TM6740ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<Pulnix::TM6740ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Pulnix::TM6740ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Quartz::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Quartz::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Quartz::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Rayonix::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Rayonix::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Rayonix::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Rayonix::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<Rayonix::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Rayonix::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Timepix::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<Timepix::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Timepix::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Timepix::ConfigV2)) {
    if (inEvt) {
      boost::shared_ptr<Timepix::ConfigV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Timepix::ConfigV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Timepix::ConfigV3)) {
    if (inEvt) {
      boost::shared_ptr<Timepix::ConfigV3> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Timepix::ConfigV3> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Timepix::DataV1)) {
    if (inEvt) {
      boost::shared_ptr<Timepix::DataV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Timepix::DataV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(Timepix::DataV2)) {
    if (inEvt) {
      boost::shared_ptr<Timepix::DataV2> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<Timepix::DataV2> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(UsdUsb::ConfigV1)) {
    if (inEvt) {
      boost::shared_ptr<UsdUsb::ConfigV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<UsdUsb::ConfigV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
  if (keyCppType == typeid(UsdUsb::DataV1)) {
    if (inEvt) {
      boost::shared_ptr<UsdUsb::DataV1> p = evt.get(eventKey.src(), eventKey.key());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from evt");
      if (p) {
        cout << "EventKey src: " << src << " type: ";
        PrintPsddl::print(*p);
      }
    } else {
      boost::shared_ptr<UsdUsb::DataV1> p = env.configStore().get(eventKey.src());
      if (not p) MsgLog(logger,error,"Did not get object for " << eventKey << " from configStore");
      if (p) {
        PrintPsddl::print(*p);
        cout << "EventKey src: " << src << " type: ";
      }
    }
    return;
  }
}

} // PsanaTools namespace
