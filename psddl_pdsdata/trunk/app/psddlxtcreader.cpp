#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"
#include "pdsdata/xtc/XtcIterator.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"

#include "psddl_pdsdata/acqiris.ddl.h"
#include "psddl_pdsdata/bld.ddl.h"
#include "psddl_pdsdata/camera.ddl.h"
#include "psddl_pdsdata/control.ddl.h"
#include "psddl_pdsdata/cspad.ddl.h"
#include "psddl_pdsdata/encoder.ddl.h"
#include "psddl_pdsdata/epics.ddl.h"
#include "psddl_pdsdata/evr.ddl.h"
#include "psddl_pdsdata/fccd.ddl.h"
#include "psddl_pdsdata/ipimb.ddl.h"
#include "psddl_pdsdata/lusi.ddl.h"
#include "psddl_pdsdata/opal1k.ddl.h"
#include "psddl_pdsdata/pnccd.ddl.h"
#include "psddl_pdsdata/princeton.ddl.h"
#include "psddl_pdsdata/pulnix.ddl.h"

using namespace PsddlPds;

static unsigned eventCount = 0;

using namespace Pds;

class myLevelIter : public XtcIterator {
public:
  enum {Stop, Continue};
  myLevelIter(Xtc* xtc, unsigned depth, long long int lliOffset) : XtcIterator(xtc), _depth(depth), _lliOffset(lliOffset) {}

  void process(const DetInfo& d, const Camera::FrameV1& f) {
    printf("*** Processing Camera::FrameV1 object\n");
  }
  void process(const DetInfo&, const Acqiris::DataDescV1&) {
    printf("*** Processing Acqiris::DataDescV1 object\n");
  }
  void process(const DetInfo&, const Acqiris::ConfigV1&) {
    printf("*** Processing Acqiris::ConfigV1 object\n");
  }
  void process(const DetInfo&, const Ipimb::DataV1&) {
    printf("*** Processing Ipimb::DataV1 object\n");
  }
  void process(const DetInfo&, const Ipimb::DataV2&) {
    printf("*** Processing Ipimb::DataV2 object\n");
  }
  void process(const DetInfo&, const Ipimb::ConfigV1&) {
    printf("*** Processing Ipimb::ConfigV1 object\n");
  }
  void process(const DetInfo&, const Ipimb::ConfigV2&) {
    printf("*** Processing Ipimb::ConfigV2 object\n");
  }
  void process(const DetInfo&, const Encoder::DataV1&) {
    printf("*** Processing Encoder::DataV1 object\n");
  }
  void process(const DetInfo&, const Encoder::DataV2&) {
    printf("*** Processing Encoder::DataV2 object\n");
  }
  void process(const DetInfo&, const Encoder::ConfigV1&) {
    printf("*** Processing Encoder::ConfigV1 object\n");
  }
  void process(const DetInfo&, const Opal1k::ConfigV1&) {
    printf("*** Processing Opal1k::ConfigV1 object\n");
  }
  void process(const DetInfo&, const Pulnix::TM6740ConfigV1&) {
    printf("*** Processing Pulnix::TM6740ConfigV1 object\n");
  }
  void process(const DetInfo&, const Camera::FrameFexConfigV1&) {
    printf("*** Processing Camera::FrameFexConfigV1 object\n");
  }
  void process(const DetInfo&, const FCCD::FccdConfigV1&) {
    printf("*** Processing FCCD::FccdConfigV1 object\n");
  }
  void process(const DetInfo&, const FCCD::FccdConfigV2&) {
    printf("*** Processing FCCD::FccdConfigV2 object\n");
  }
  void process(const DetInfo&, const Camera::TwoDGaussianV1& o) {
    printf("*** Processing Camera::TwoDGaussianV1 object\n");
  }
  void process(const DetInfo& det, const PNCCD::ConfigV1& config) {
    if ( det.detId() != 0 )
    {
      printf( "myLevelIter::process(...,PNCCD::ConfigV1&): pnCCD detector Id (%d) is not 0\n", det.detId() );
      return;
    }
    if ( det.devId() < 0 || det.devId() > 1)
    {
      printf( "myLevelIter::process(...,PNCCD::ConfigV1&): pnCCD device Id (%d) is out of range (0..1)\n", det.devId() );
      return;
    }
    
    _pnCcdCfgListV1[det.devId()] = config;
    printf("*** Processing pnCCD config.  Number of Links: %d, PayloadSize per Link: %d\n",
           config.numLinks(),config.payloadSizePerLink());
  }  
  void process(const DetInfo& det, const PNCCD::ConfigV2& config) {
    if ( det.detId() != 0 )
    {
      printf( "myLevelIter::process(...,PNCCD::ConfigV2&): pnCCD detector Id (%d) is not 0\n", det.detId() );
      return;
    }
    if ( det.devId() < 0 || det.devId() > 1)
    {
      printf( "myLevelIter::process(...,PNCCD::ConfigV2&): pnCCD device Id (%d) is out of range (0..1)\n", det.devId() );
      return;
    }

    _pnCcdCfgListV2[det.devId()] = config;
    printf("*** Processing pnCCD config.  Number of Links: %u, PayloadSize per Link: %u\n",
           config.numLinks(),config.payloadSizePerLink());
    printf("\tNumber of Channels %u, Number of Rows %u, Number of SubModule Channels %u\n\tNumber of SubModule Rows %u, Number of SubModules, %u\n",
        config.numChannels(),config.numRows(), config.numSubmoduleChannels(),config.numSubmoduleRows(),config.numSubmodules());
    printf("\tCamex Magic 0x%x, info %s, Timing File Name %s\n", config.camexMagic(),config.info(),config.timingFName());
  }
  void process(const DetInfo& det, const PNCCD::FrameV1* f) {
    if ( det.detId() != 0 )
    {
      printf( "myLevelIter::process(...,PNCCD::FrameV1*): pnCCD detector Id (%d) is not 0\n", det.detId() );
      return;
    }
    if ( det.devId() < 0 || det.devId() > 1)
    {
      printf( "myLevelIter::process(...,PNCCD::FrameV1*): pnCCD device Id (%d) is out of range (0..1)\n", det.devId() );
      return;
    }
    
    printf("*** Processing pnCCD Frame\n");
  }  
  void process(const DetInfo&, const ControlData::ConfigV1& config) {
    printf("*** Processing Control config object\n");    
    
    printf( "Control PV Number = %d, Monitor PV Number = %d\n", config.npvControls(), config.npvMonitors() );
    for(unsigned int iPvControl=0; iPvControl < config.npvControls(); iPvControl++) {      
      const ControlData::PVControl& pvControlCur = config.pvControls(iPvControl);
      if (pvControlCur.array())
        printf( "%s[%d] = ", pvControlCur.name(), pvControlCur.index() );
      else
        printf( "%s = ", pvControlCur.name() );
      printf( "%lf\n", pvControlCur.value() );
    }
    
    for(unsigned int iPvMonitor=0; iPvMonitor < config.npvMonitors(); iPvMonitor++) {      
      const ControlData::PVMonitor& pvMonitorCur = config.pvMonitors(iPvMonitor);
      if (pvMonitorCur.array())
        printf( "%s[%d]  ", pvMonitorCur.name(), pvMonitorCur.index() );
      else
        printf( "%s  ", pvMonitorCur.name() );
      printf( "Low %lf  High %lf\n", pvMonitorCur.loValue(), pvMonitorCur.hiValue() );
    }
          
  }  
  void process(const DetInfo&, const Epics::EpicsPvHeader& epicsPv)
  {    
    printf("*** Processing Epics object\n");
    epicsPv.print();
    printf( "\n" );
  }
  void process(const DetInfo&, const Bld::BldDataFEEGasDetEnergy& bldData) {
    printf("*** Processing FEEGasDetEnergy object\n");
    printf("GDET:FEE:11:ENRC ( in mJ ): %lf\n", bldData.f_11_ENRC() );    
    printf("GDET:FEE:12:ENRC ( in mJ ): %lf\n", bldData.f_12_ENRC() );
    printf("GDET:FEE:21:ENRC ( in mJ ): %lf\n", bldData.f_21_ENRC() ); 
    printf("GDET:FEE:22:ENRC ( in mJ ): %lf\n", bldData.f_22_ENRC() );
    printf( "\n" );    
  }  
  void process(const DetInfo&, const Bld::BldDataEBeamV0& bldData) {
    printf("*** Processing EBeamV0 object\n");
    printf( "ebeamCharge   ( in nC )  : %lf\n", bldData.ebeamCharge()  ); 
    printf( "ebeamL3Energy ( in MeV ) : %lf\n", bldData.ebeamL3Energy()); 
    printf( "ebeamLTUPosX  ( in mm )  : %lf\n", bldData.ebeamLTUPosX() ); 
    printf( "ebeamLTUPosY  ( in mm )  : %lf\n", bldData.ebeamLTUPosY() ); 
    printf( "ebeamLTUAngX  ( in mrad ): %lf\n", bldData.ebeamLTUAngX() ); 
    printf( "ebeamLTUAngY  ( in mrad ): %lf\n", bldData.ebeamLTUAngY() );
    printf( "\n" );    
  }  
  void process(const DetInfo&, const Bld::BldDataEBeamV1& bldData) {
    printf("*** Processing EBeamV1 object\n");
    printf( "ebeamCharge   ( in nC )  : %lf\n", bldData.ebeamCharge()  ); 
    printf( "ebeamL3Energy ( in MeV ) : %lf\n", bldData.ebeamL3Energy()); 
    printf( "ebeamLTUPosX  ( in mm )  : %lf\n", bldData.ebeamLTUPosX() ); 
    printf( "ebeamLTUPosY  ( in mm )  : %lf\n", bldData.ebeamLTUPosY() ); 
    printf( "ebeamLTUAngX  ( in mrad ): %lf\n", bldData.ebeamLTUAngX() ); 
    printf( "ebeamLTUAngY  ( in mrad ): %lf\n", bldData.ebeamLTUAngY() );
    printf( "ebeamPkCurrBC2( in mA )  : %lf\n", bldData.ebeamPkCurrBC2() );
    printf( "\n" );    
  }  
  void process(const DetInfo&, const Bld::BldDataEBeamV2& bldData) {
    printf("*** Processing EBeamV2 object\n");
    printf( "ebeamCharge   ( in nC )  : %lf\n", bldData.ebeamCharge()  ); 
    printf( "ebeamL3Energy ( in MeV ) : %lf\n", bldData.ebeamL3Energy()); 
    printf( "ebeamLTUPosX  ( in mm )  : %lf\n", bldData.ebeamLTUPosX() ); 
    printf( "ebeamLTUPosY  ( in mm )  : %lf\n", bldData.ebeamLTUPosY() ); 
    printf( "ebeamLTUAngX  ( in mrad ): %lf\n", bldData.ebeamLTUAngX() ); 
    printf( "ebeamLTUAngY  ( in mrad ): %lf\n", bldData.ebeamLTUAngY() );
    printf( "ebeamPkCurrBC2( in mA )  : %lf\n", bldData.ebeamPkCurrBC2() );
    printf( "ebeamEnergyBC2( in mA )  : %lf\n", bldData.ebeamEnergyBC2() );
    printf( "\n" );    
  }  
  void process(const DetInfo&, const Bld::BldDataPhaseCavity& bldData) {
    printf("*** Processing PhaseCavity object\n");
    printf("FitTime1 ( in pico-seconds ): %lf\n", bldData.fitTime1() );
    printf("FitTime2 ( in pico-seconds ): %lf\n", bldData.fitTime2() );
    printf("Charge1  ( in pico-columbs ): %lf\n", bldData.charge1() );
    printf("Charge2  ( in pico-columbs ): %lf\n", bldData.charge2() );
    printf( "\n" );    
  } 
  void process(const DetInfo&, const Bld::BldDataIpimbV0& bldData) {
    printf("*** Processing Bld-Ipimb object\n");
    printf("BLD Shared IPIMB Data:\n");
    unsigned long long int counter = bldData.ipimbData().triggerCounter();
    printf("  Trig Count      : %llu \n",counter);
    printf("  IpimbDataCh-0   : %f   \n",bldData.ipimbData().channel0Volts());
    printf("  IpimbDataCh-1   : %f   \n",bldData.ipimbData().channel1Volts());            
    printf("  IpimbDataCh-2   : %f   \n",bldData.ipimbData().channel2Volts());    
    printf("  IpimbDataCh-3   : %f   \n",bldData.ipimbData().channel3Volts());

    printf("  IpmFexDataCh-0  : %f   \n",bldData.ipmFexData().channel()[0]);        
    printf("  IpmFexDataCh-1  : %f   \n",bldData.ipmFexData().channel()[1]);
    printf("  IpmFexDataCh-2  : %f   \n",bldData.ipmFexData().channel()[2]);
    printf("  IpmFexDataCh-3  : %f   \n",bldData.ipmFexData().channel()[3]);
    printf("  IpmFexDataSum   : %f   \n",bldData.ipmFexData().sum());
    printf("  IpmFexDataXpos  : %f   \n",bldData.ipmFexData().xpos());
    printf("  IpmFexDataYpos  : %f   \n",bldData.ipmFexData().ypos());
    printf( "\n" );    
  }   
  void process(const DetInfo&, const Bld::BldDataIpimbV1& bldData) {
    printf("*** Processing Bld-Ipimb object\n");
    printf("BLD Shared IPIMB Data:\n");
    unsigned long long int counter = bldData.ipimbData().triggerCounter();
    printf("  Trig Count      : %llu \n",counter);
    printf("  IpimbDataCh-0   : %f   \n",bldData.ipimbData().channel0Volts());
    printf("  IpimbDataCh-1   : %f   \n",bldData.ipimbData().channel1Volts());            
    printf("  IpimbDataCh-2   : %f   \n",bldData.ipimbData().channel2Volts());    
    printf("  IpimbDataCh-3   : %f   \n",bldData.ipimbData().channel3Volts());
    printf("  IpimbDataCh-0ps : %f   \n",bldData.ipimbData().channel0psVolts());
    printf("  IpimbDataCh-1ps : %f   \n",bldData.ipimbData().channel1psVolts());            
    printf("  IpimbDataCh-2ps : %f   \n",bldData.ipimbData().channel2psVolts());    
    printf("  IpimbDataCh-3ps : %f   \n",bldData.ipimbData().channel3psVolts());

    printf("  IpmFexDataCh-0  : %f   \n",bldData.ipmFexData().channel()[0]);        
    printf("  IpmFexDataCh-1  : %f   \n",bldData.ipmFexData().channel()[1]);
    printf("  IpmFexDataCh-2  : %f   \n",bldData.ipmFexData().channel()[2]);
    printf("  IpmFexDataCh-3  : %f   \n",bldData.ipmFexData().channel()[3]);
    printf("  IpmFexDataSum   : %f   \n",bldData.ipmFexData().sum());
    printf("  IpmFexDataXpos  : %f   \n",bldData.ipmFexData().xpos());
    printf("  IpmFexDataYpos  : %f   \n",bldData.ipmFexData().ypos());
    printf( "\n" );    
  }   
  void process(const DetInfo&, const EvrData::IOConfigV1&) {
    printf("*** Processing EVR IOconfig V1 object\n");
  }
  void process(const DetInfo&, const EvrData::ConfigV1&) {
    printf("*** Processing EVR config V1 object\n");
  }
  void process(const DetInfo&, const EvrData::ConfigV2&) {
    printf("*** Processing EVR config V2 object\n");
  }
  void process(const DetInfo&, const EvrData::ConfigV3&) {
    printf("*** Processing EVR config V3 object\n");
  }
  void process(const DetInfo&, const EvrData::ConfigV4&) {
    printf("*** Processing EVR config V4 object\n");
  }
  void process(const DetInfo&, const EvrData::DataV3& data) {
    printf("*** Processing Evr Data object\n");
    eventCount++;    

    printf( "# of Fifo Events: %u\n", data.numFifoEvents() );
    for ( unsigned int iEventIndex=0; iEventIndex< data.numFifoEvents(); iEventIndex++ ) {
      const EvrData::FIFOEvent& event = data.fifoEvents(iEventIndex);
      //printf( "[%02u] Event Code %u  TimeStampHigh 0x%x  TimeStampLow 0x%x\n",
      //  iEventIndex, event.EventCode, event.TimestampHigh, event.TimestampLow );
      if (event.eventCode() == 162)
        printf ("Blank shot eventcode 162 found at eventNo: %u \n",eventCount); 
    }    
    printf( "\n" );    
  }  
  void process(const DetInfo&, const Princeton::ConfigV1&) {
    printf("*** Processing Princeton ConfigV1 object\n");
  }
  void process(const DetInfo&, const Princeton::FrameV1&) {
    printf("*** Processing Princeton FrameV1 object\n");
  }
  void process(const DetInfo&, const Princeton::InfoV1&) {
    printf("*** Processing Princeton InfoV1 object\n");
  }
  void process(const DetInfo&, const CsPad::ElementV1&) {
    printf("*** Processing CsPad ElementV1 object\n");
  }
  void process(const DetInfo&, const CsPad::ElementV2&) {
    printf("*** Processing CsPad ElementV2 object\n");
  }
  void process(const DetInfo&, const CsPad::ConfigV1&) {
    printf("*** Processing CsPad ConfigV1 object\n");
  }
  void process(const DetInfo&, const CsPad::ConfigV2&) {
    printf("*** Processing CsPad ConfigV2 object\n");
  }
  void process(const DetInfo&, const CsPad::ConfigV3&) {
    printf("*** Processing CsPad ConfigV3 object\n");
  }
  void process(const DetInfo&, const Lusi::IpmFexConfigV1&) {
    printf("*** Processing LUSI IpmFexConfigV1 object\n");
  }
  void process(const DetInfo&, const Lusi::IpmFexConfigV2&) {
    printf("*** Processing LUSI IpmFexConfigV2 object\n");
  }
  void process(const DetInfo&, const Lusi::IpmFexV1&) {
    printf("*** Processing LUSI IpmFexV1 object\n");
  }
  void process(const DetInfo&, const Lusi::DiodeFexConfigV1&) {
    printf("*** Processing LUSI DiodeFexConfigV1 object\n");
  }
  void process(const DetInfo&, const Lusi::DiodeFexConfigV2&) {
    printf("*** Processing LUSI DiodeFexConfigV2 object\n");
  }
  void process(const DetInfo&, const Lusi::DiodeFexV1&) {
    printf("*** Processing LUSI DiodeFexV1 object\n");
  }
  int process(Xtc* xtc) {
    unsigned      i         =_depth; while (i--) printf("  ");
    Level::Type   level     = xtc->src.level();
    printf("%s level  offset %Ld (0x%Lx), payload size %d contains: %s: ",
      Level::name(level), _lliOffset, _lliOffset, xtc->sizeofPayload(), TypeId::name(xtc->contains.id()));
    long long lliOffsetPayload = _lliOffset + sizeof(Xtc);
    _lliOffset += sizeof(Xtc) + xtc->sizeofPayload();
     
    const DetInfo& info = *(DetInfo*)(&xtc->src);
    if (level==Level::Source) {
      printf("%s,%d  %s,%d\n",
             DetInfo::name(info.detector()),info.detId(),
             DetInfo::name(info.device()),info.devId());
    } else {
      ProcInfo& info = *(ProcInfo*)(&xtc->src);
      printf("IpAddress 0x%x ProcessId 0x%x\n",info.ipAddr(),info.processId());
    }
    if (level < 0 || level >= Level::NumberOfLevels )
    {
        printf("Unsupported Level %d\n", (int) level);
        return Continue;
    }    
    switch (xtc->contains.id()) {
    case (TypeId::Id_Xtc) : {
      myLevelIter iter(xtc,_depth+1, lliOffsetPayload);
      iter.iterate();
      break;
    }
    case (TypeId::Id_Frame) :
      process(info, *(const Camera::FrameV1*)(xtc->payload()));
      break;
    case (TypeId::Id_AcqWaveform) :
      process(info, *(const Acqiris::DataDescV1*)(xtc->payload()));
      break;
    case (TypeId::Id_AcqConfig) :
    {      
      unsigned version = xtc->contains.version();
      switch (version) {
      case 1:
        process(info,*(const Acqiris::ConfigV1*)(xtc->payload()));
        break;
      default:
        printf("Unsupported acqiris configuration version %d\n",version);
        break;
      }
      break;      
    }
    case (TypeId::Id_IpimbData) :
    {
      switch (xtc->contains.version()) {
      case 1:
        process(info, *(const Ipimb::DataV1*)(xtc->payload()));
        break;
      case 2:
        process(info, *(const Ipimb::DataV2*)(xtc->payload()));
        break;
      default:
        printf("Unsupported ipimb data version %d\n",xtc->contains.version());
        break;
      }
      break;      
      break;
    }
    case (TypeId::Id_IpimbConfig) :
    {      
      switch (xtc->contains.version()) {
      case 1:
        process(info,*(const Ipimb::ConfigV1*)(xtc->payload()));
        break;
      case 2:
        process(info,*(const Ipimb::ConfigV2*)(xtc->payload()));
        break;
      default:
        printf("Unsupported ipimb configuration version %d\n",xtc->contains.version());
        break;
      }
      break;      
    }
    case (TypeId::Id_EncoderData) :
    {      
      unsigned version = xtc->contains.version();
      switch (version) {
      case 1:
        process(info,*(const Encoder::DataV1*)(xtc->payload()));
        break;
      case 2:
        process(info,*(const Encoder::DataV2*)(xtc->payload()));
        break;
      default:
        printf("Unsupported encoder data version %d\n",version);
        break;
      }
      break;      
    }
    case (TypeId::Id_EncoderConfig) :
    {      
      unsigned version = xtc->contains.version();
      switch (version) {
      case 1:
        process(info,*(const Encoder::ConfigV1*)(xtc->payload()));
        break;
      default:
        printf("Unsupported encoder configuration version %d\n",version);
        break;
      }
      break;      
    }
    case (TypeId::Id_TwoDGaussian) :
      process(info, *(const Camera::TwoDGaussianV1*)(xtc->payload()));
      break;
    case (TypeId::Id_Opal1kConfig) :
      process(info, *(const Opal1k::ConfigV1*)(xtc->payload()));
      break;
    case (TypeId::Id_FrameFexConfig) :
      process(info, *(const Camera::FrameFexConfigV1*)(xtc->payload()));
      break;
    case (TypeId::Id_pnCCDconfig) :
      {
      unsigned version = xtc->contains.version();
      switch (version) {
        case 1:
          process(info, *(const PNCCD::ConfigV1*)(xtc->payload()));
          break;
        case 2:
          process(info, *(const PNCCD::ConfigV2*)(xtc->payload()));
          break;
        default:
          printf("Unsupported pnCCD configuration version %d\n",version);
      }
      break;
      }
    case (TypeId::Id_pnCCDframe) :
      {
      process(info, (const PNCCD::FrameV1*)(xtc->payload()));
      break;
      }
    case (TypeId::Id_EvrIOConfig) :
      {      
      process(info, *(const EvrData::IOConfigV1*)(xtc->payload()));
      break;
      }
    case (TypeId::Id_EvrConfig) :
    {      
      unsigned version = xtc->contains.version();
      switch (version) {
      case 1:
        process(info, *(const EvrData::ConfigV1*)(xtc->payload()));
        break;
      case 2:
        process(info, *(const EvrData::ConfigV2*)(xtc->payload()));
        break;
      case 3:
        process(info, *(const EvrData::ConfigV3*)(xtc->payload()));
        break;
      case 4:
        process(info, *(const EvrData::ConfigV4*)(xtc->payload()));
        break;
      default:
        printf("Unsupported evr configuration version %d\n",version);
        break;
      }
      break;      
    }
    case (TypeId::Id_EvrData) :
    {
      process(info, *(const EvrData::DataV3*) xtc->payload() );
      break;        
    }
    case (TypeId::Id_ControlConfig) :
      process(info, *(const ControlData::ConfigV1*)(xtc->payload()));
      break;
    case (TypeId::Id_Epics) :      
    {
      int iVersion = xtc->contains.version();
      if ( iVersion != Epics::iXtcVersion ) 
      {
          printf( "Xtc Epics version (%d) is not compatible with reader supported version (%d)", iVersion, Epics::iXtcVersion );
          break;
      }
      process(info, *(const Epics::EpicsPvHeader*)(xtc->payload()));
      break;
    }
    /*
     * BLD data
     */
    case (TypeId::Id_FEEGasDetEnergy) :
    {
      process(info, *(const Bld::BldDataFEEGasDetEnergy*) xtc->payload() );
      break;        
    }
    case (TypeId::Id_EBeam) :
    {
      switch(xtc->contains.version()) {
      case 0:
        process(info, *(const Bld::BldDataEBeamV0*) xtc->payload() );
        break; 
      case 1:
        process(info, *(const Bld::BldDataEBeamV1*) xtc->payload() );
        break; 
      case 2:
        process(info, *(const Bld::BldDataEBeamV2*) xtc->payload() );
        break; 
      default:
        break;
      }       
    }    
    case (TypeId::Id_PhaseCavity) :
    {
      process(info, *(const Bld::BldDataPhaseCavity*) xtc->payload() );
      break;        
    }
    case (TypeId::Id_SharedIpimb) :
    {
      switch(xtc->contains.version()) {
      case 0:
        process(info, *(const Bld::BldDataIpimbV0*) xtc->payload() );
        break; 
      case 1:
        process(info, *(const Bld::BldDataIpimbV1*) xtc->payload() );
        break; 
      default:
        break;
      }       
      break;        
    }	
    case (TypeId::Id_PrincetonConfig) :
    {
      process(info, *(const Princeton::ConfigV1*)(xtc->payload()));
      break;
    }
    case (TypeId::Id_PrincetonFrame) :
    {
      process(info, *(const Princeton::FrameV1*)(xtc->payload()));
      break;
    }    
    case (TypeId::Id_PrincetonInfo) :
    {
      process(info, *(const Princeton::InfoV1*)(xtc->payload()));
      break;
    }    
    case (TypeId::Id_CspadElement) :
    {
      switch(xtc->contains.version()) {
      case 1:
        process(info, *(const CsPad::ElementV1*)(xtc->payload()));
        break; 
      case 2:
        process(info, *(const CsPad::ElementV2*)(xtc->payload()));
        break; 
      default:
        break;
      }       
      break;
    }    
    case (TypeId::Id_CspadConfig) :
    {
      switch(xtc->contains.version()) {
      case 1:
        process(info, *(const CsPad::ConfigV1*)(xtc->payload()));
        break; 
      case 2:
        process(info, *(const CsPad::ConfigV2*)(xtc->payload()));
        break; 
      case 3:
        process(info, *(const CsPad::ConfigV3*)(xtc->payload()));
        break; 
      default:
        break;
      }       
      break;
    }    
    case (TypeId::Id_IpmFexConfig) :
    {
      switch(xtc->contains.version()) {
      case 1:
        process(info, *(const Lusi::IpmFexConfigV1*)(xtc->payload()));
        break; 
      case 2:
        process(info, *(const Lusi::IpmFexConfigV2*)(xtc->payload()));
        break; 
      default:
        break;
      }       
      break;
    }    
    case (TypeId::Id_IpmFex) :
    {
      process(info, *(const Lusi::IpmFexV1*)(xtc->payload()));
      break;
    }    
    case (TypeId::Id_DiodeFexConfig) :
    {
      switch(xtc->contains.version()) {
      case 1:
        process(info, *(const Lusi::DiodeFexConfigV1*)(xtc->payload()));
        break; 
      case 2:
        process(info, *(const Lusi::DiodeFexConfigV2*)(xtc->payload()));
        break; 
      default:
        break;
      }       
      break;
    }    
    case (TypeId::Id_DiodeFex) :
    {
      process(info, *(const Lusi::DiodeFexV1*)(xtc->payload()));
      break;
    }    
    default :
      printf("Unsupported TypeId %d\n", (int) xtc->contains.id());
      break;
    }
    return Continue;
  }
private:
  unsigned       _depth;
  long long int  _lliOffset;

  /* static private data */
  static PNCCD::ConfigV1 _pnCcdCfgListV1[2];
  static PNCCD::ConfigV2 _pnCcdCfgListV2[2];
};

PNCCD::ConfigV1 myLevelIter::_pnCcdCfgListV1[2] = { PNCCD::ConfigV1(), PNCCD::ConfigV1() };
PNCCD::ConfigV2 myLevelIter::_pnCcdCfgListV2[2] = { PNCCD::ConfigV2(), PNCCD::ConfigV2() };

void usage(char* progname) {
  fprintf(stderr,"Usage: %s -f <filename> [-h]\n", progname);
}

int main(int argc, char* argv[]) {
  int c;
  char* xtcname=0;
  int parseErr = 0;

  while ((c = getopt(argc, argv, "hf:")) != -1) {
    switch (c) {
    case 'h':
      usage(argv[0]);
      exit(0);
    case 'f':
      xtcname = optarg;
      break;
    default:
      parseErr++;
    }
  }
  
  if (!xtcname) {
    usage(argv[0]);
    exit(2);
  }

  int fd = open(xtcname, O_RDONLY | O_LARGEFILE);
  if (fd < 0) {
    perror("Unable to open file %s\n");
    exit(2);
  }

  XtcFileIterator iter(fd,0x900000);
  Dgram* dg;
  long long int lliOffset = lseek64( fd, 0, SEEK_CUR );  
  while ((dg = iter.next())) {
    printf("%s transition: time 0x%x/0x%x, offset %Ld (0x%Lx), payloadSize %d\n",TransitionId::name(dg->seq.service()),
           dg->seq.stamp().fiducials(),dg->seq.stamp().ticks(), lliOffset, lliOffset, dg->xtc.sizeofPayload());
    myLevelIter iter(&(dg->xtc),0, lliOffset + sizeof(Xtc) + sizeof(*dg) - sizeof(dg->xtc));
    iter.iterate();
    lliOffset = lseek64( fd, 0, SEEK_CUR ); // get the file offset for the next iteration
  }

  ::close(fd);
  return 0;
}
