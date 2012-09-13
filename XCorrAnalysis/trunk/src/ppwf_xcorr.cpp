//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ppwf_xcorr...
//
// Author List:
//      Ingrid Ofte
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XCorrAnalysis/ppwf_xcorr.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"

// Include detector data headers from psddl_psana package:
#include "psddl_psana/evr.ddl.h"
#include "psddl_psana/encoder.ddl.h"
#include "psddl_psana/bld.ddl.h"
#include "psddl_psana/ipimb.ddl.h"
#include "psddl_psana/fccd.ddl.h"
#include "psddl_psana/opal1k.ddl.h"
#include "psddl_psana/camera.ddl.h"
#include "psddl_psana/acqiris.ddl.h"

#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace XCorrAnalysis;
PSANA_MODULE_FACTORY(ppwf_xcorr)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XCorrAnalysis {

//----------------
// Constructors --
//----------------
ppwf_xcorr::ppwf_xcorr (const std::string& name)
  : Module(name)
  , m_evrSrc()
  , m_encSrc()
  , m_pcSrc()
  , m_ipmSrc()
  , m_fccdSrc()
  , m_opalSrc()
  , m_maxEvents()
  , m_filter()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_evrSrc = configStr("evrSrc","DetInfo(:Evr)");
  m_encSrc = configStr("encSrc","DetInfo(:Encoder)");
  m_pcSrc = configStr("pcSrc", "BldInfo(PhaseCavity)");
  m_ipmSrc = configStr("ipmSrc", "DetInfo(:Ipimb)");
  m_fccdSrc = configStr("fccdSrc", "DetInfo(:Fccd)");
  m_opalSrc = configStr("opalSrc", "DetInfo(:Opal1000)");  
  m_acqSrc = configStr("acqSrc", "DetInfo(:Acqiris)");
  m_maxEvents = config("events", 32U);
  m_filter = config("filter", false);
}

//--------------
// Destructor --
//--------------
ppwf_xcorr::~ppwf_xcorr ()
{
}

/// Method which is called once at the beginning of the job
void 
ppwf_xcorr::beginJob(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the run
void 
ppwf_xcorr::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
ppwf_xcorr::beginCalibCycle(Event& evt, Env& env)
{
  shared_ptr<Psana::Opal1k::ConfigV1> config = env.configStore().get(m_opalSrc);
  if (config.get()) {
    
    WithMsgLog(name(), info, str) {
      str << "Psana::Opal1k::ConfigV1:";
      str << "\n  black_level = " << config->black_level();
      str << "\n  gain_percent = " << config->gain_percent();
      str << "\n  output_resolution = " << config->output_resolution();
      str << "\n  vertical_binning = " << config->vertical_binning();
      str << "\n  output_mirroring = " << config->output_mirroring();
      str << "\n  vertical_remapping = " << int(config->vertical_remapping());
      str << "\n  output_offset = " << config->output_offset();
      str << "\n  output_resolution_bits = " << config->output_resolution_bits();
      str << "\n  defect_pixel_correction_enabled = " << int(config->defect_pixel_correction_enabled());
      str << "\n  output_lookup_table_enabled = " << int(config->output_lookup_table_enabled());

      if (config->output_lookup_table_enabled()) {
        const ndarray<uint16_t, 1>& output_lookup_table = config->output_lookup_table();
        str << "\n  output_lookup_table =";
        for (unsigned i = 0; i < output_lookup_table.size(); ++ i) {
          str << ' ' << output_lookup_table[i];
        }

      }


      if (config->number_of_defect_pixels()) {
        str << "\n  defect_pixel_coordinates =";
        const ndarray<Psana::Camera::FrameCoord, 1>& coord = config->defect_pixel_coordinates();
        for (unsigned i = 0; i < coord.size(); ++ i) {
	  str << "(" << coord[i].column() << ", " << coord[i].row() << ")";
        }
      }
    }
  }

  shared_ptr<Psana::FCCD::FccdConfigV2> config2 = env.configStore().get(m_fccdSrc);
  if (config2.get()) {
    
    WithMsgLog(name(), info, str) {
      str << "FCCD::FccdConfigV2:";
      str << "\n  outputMode = " << config2->outputMode();
      str << "\n  ccdEnable = " << int(config2->ccdEnable());
      str << "\n  focusMode = " << int(config2->focusMode());
      str << "\n  exposureTime = " << config2->exposureTime();
      str << "\n  dacVoltages = [" << config2->dacVoltages()[0]
          << " " << config2->dacVoltages()[1] << " ...]";
      str << "\n  waveforms = [" << config2->waveforms()[0]
          << " " << config2->waveforms()[1] << " ...]";
      str << "\n  width = " << config2->width();
      str << "\n  height = " << config2->height();
      str << "\n  trimmedWidth = " << config2->trimmedWidth();
      str << "\n  trimmedHeight = " << config2->trimmedHeight();
    }

  }
  
  shared_ptr<Psana::Acqiris::ConfigV1> acqConfig = env.configStore().get(m_acqSrc);
  if (acqConfig.get()) {
    WithMsgLog(name(), info, str) {
      str << "Acqiris::ConfigV1: nbrBanks=" << acqConfig->nbrBanks()
          << " channelMask=" << acqConfig->channelMask()
          << " nbrChannels=" << acqConfig->nbrChannels()
          << " nbrConvertersPerChannel=" << acqConfig->nbrConvertersPerChannel();

      const Psana::Acqiris::HorizV1& h = acqConfig->horiz();
      str << "\n  horiz: sampInterval=" << h.sampInterval()
	  << " delayTime=" << h.delayTime()
	  << " nbrSegments=" << h.nbrSegments()
	  << " nbrSamples=" << h.nbrSamples();
      
      const ndarray<Psana::Acqiris::VertV1, 1>& vert = acqConfig->vert();
      for (unsigned ch = 0; ch < vert.shape()[0]; ++ ch) {
        const Psana::Acqiris::VertV1& v = vert[ch];
        str << "\n  vert(" << ch << "):"
            << " fullScale=" << v.fullScale()
            << " slope=" << v.slope()
            << " offset=" << v.offset()
            << " coupling=" << v.coupling()
            << " bandwidth=" << v.bandwidth();
      }
    }
  }


  
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
ppwf_xcorr::event(Event& evt, Env& env)
{
  // tis is how to skip event (all downstream modules will not be called)
  if (m_filter && m_count % 10 == 0) skip();
  
  // this is how to gracefully stop analysis job
  if (m_count >= m_maxEvents) stop();


  // example of getting non-detector data from event
  shared_ptr<PSEvt::EventId> eventId = evt.get();
  if (eventId.get()) {
    // example of producing messages using MgsLog facility
    MsgLog(name(), info, "event ID: " << *eventId);
  }

  // *** EVR data (I don't know if this is ever used?)
  shared_ptr<Psana::EvrData::DataV3> myEvrData = evt.get(m_evrSrc);
  if (myEvrData.get()) {
    
    WithMsgLog(name(), info, str) {
      str << " numFifoEvents=" << myEvrData->numFifoEvents() << "\n" ;
      ndarray<Psana::EvrData::FIFOEvent, 1>  fifoEvents = myEvrData->fifoEvents();
      for (unsigned i(0); i<myEvrData->numFifoEvents(); i++){
	str<< " EventCode =  " << fifoEvents[i].eventCode() ; 
	str<< " Timestamp low =  "  << fifoEvents[i].timestampLow() ; 
	str<< " Timestamp high =  " << fifoEvents[i].timestampHigh() << "\n"; 
      }
    }
  }

  //  *** Get Encoder to determine DelayTime and BinNumber
  shared_ptr<Psana::Encoder::DataV2> myEncoder = evt.get(m_encSrc);
  if (myEncoder.get()) {

    WithMsgLog(name(), info, str) {
      str << "Encoder::DataV2:"
          << " timestamp = " << myEncoder->timestamp()
          << " encoder_count =";
      const ndarray<uint32_t, 1>& counts = myEncoder->encoder_count();
      for (unsigned i = 0; i != counts.size(); ++ i) {
        str << " " << counts[i];
      }

    }
  }

//  *** Get Phase Cavity to determine DelayTime and BinNumber
  shared_ptr<Psana::Bld::BldDataPhaseCavity> myPhaseCavity = evt.get(m_pcSrc);
  if (myPhaseCavity.get()) {
    WithMsgLog(name(), info, str) {
      str << "Bld::BldDataPhaseCavity:"
          << "\n  fitTime1=" << myPhaseCavity->fitTime1()
          << "\n  fitTime2=" << myPhaseCavity->fitTime2()
          << "\n  charge1=" << myPhaseCavity->charge1()
          << "\n  charge2=" << myPhaseCavity->charge2();
    }
  }



// *** Retreive the Ipimb information for normalization
  shared_ptr<Psana::Ipimb::DataV2> myIpimb = evt.get(m_ipmSrc);
  if (myIpimb.get()) {

    WithMsgLog(name(), info, str) {
      str << "Ipimb::DataV2:"
          << "\n  triggerCounter = " << myIpimb->triggerCounter()
          << "\n  config = " << myIpimb->config0()
          << "," << myIpimb->config1()
          << "," << myIpimb->config2()
          << "\n  channel = " << myIpimb->channel0()
          << "," << myIpimb->channel1()
          << "," << myIpimb->channel2()
          << "," << myIpimb->channel3()
          << "\n  volts = " << myIpimb->channel0Volts()
          << "," << myIpimb->channel1Volts()
          << "," << myIpimb->channel2Volts()
          << "," << myIpimb->channel3Volts()
          << "\n  channel-ps = " << myIpimb->channel0ps()
          << "," << myIpimb->channel1ps()
          << "," << myIpimb->channel2ps()
          << "," << myIpimb->channel3ps()
          << "\n  volts-ps = " << myIpimb->channel0psVolts()
          << "," << myIpimb->channel1psVolts()
          << "," << myIpimb->channel2psVolts()
          << "," << myIpimb->channel3psVolts()
          << "\n  checksum = " << myIpimb->checksum();
    }
  }


//  Process FCCD Frame
  shared_ptr<Psana::Camera::FrameV1> fccd = evt.get(m_fccdSrc);
  if (fccd.get()) {
    //std::cout << "IOIOIOI " << fccd.depth() << std::endl;
    WithMsgLog(name(), info, str) {
      const ndarray<uint16_t, 2>& data = fccd->data16();
      str << "\n  data =";
      for (int i = 0; i < 10; ++ i) {
        str << " " << data[0][i];
      }
      str << " ...";
    }
  } else {
    std::cout << "No fccd " << std::endl;
  }

//  Process Opal Frame
  shared_ptr<Psana::Camera::FrameV1> opal = evt.get(m_opalSrc);
  if (opal.get()) {
    WithMsgLog(name(), info, str) {
      const ndarray<uint16_t, 2>& data = opal->data16();
      str << "\n  data =";
      for (int i = 0; i < 10; ++ i) {
        str << " " << data[0][i];
      }
      str << " ...";
    }
  } else {
    std::cout << "No opal " << std::endl;
  }
  
  shared_ptr<Psana::Acqiris::DataDescV1> acqData = evt.get(m_acqSrc);
  if (acqData.get()) {
    
    // find matching config object
    shared_ptr<Psana::Acqiris::ConfigV1> acqConfig = env.configStore().get(m_acqSrc);
    
    // loop over channels
    int nchan = acqData->data_shape()[0];
    for (int chan = 0; chan < nchan; ++ chan) {
      
      const Psana::Acqiris::DataDescV1Elem& elem = acqData->data(chan);

      const Psana::Acqiris::VertV1& v = acqConfig->vert()[chan];
      double slope = v.slope();
      double offset = v.offset();

      WithMsgLog(name(), info, str ) {

        str << "Acqiris::DataDescV1: channel=" << chan
           << "\n  nbrSegments=" << elem.nbrSegments()
           << "\n  nbrSamplesInSeg=" << elem.nbrSamplesInSeg()
           << "\n  indexFirstPoint=" << elem.indexFirstPoint();

        const ndarray<Psana::Acqiris::TimestampV1, 1>& timestamps = elem.timestamp();
        const ndarray<int16_t, 2>& waveforms = elem.waveforms();

        // loop over segments
        for (unsigned seg = 0; seg < elem.nbrSegments(); ++ seg) {

          str << "\n  Segment #" << seg
              << "\n    timestamp=" << timestamps[seg].pos()
              << "\n    data=[";

          unsigned size = std::min(elem.nbrSamplesInSeg(), 32U);
          for (unsigned i = 0; i < size; ++ i) {
            str << (waveforms[seg][i]*slope + offset) << ", ";
          }
          str << "...]";

        }
      }
    }
  }
  
  // increment event counter
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
ppwf_xcorr::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
ppwf_xcorr::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
ppwf_xcorr::endJob(Event& evt, Env& env)
{
}

} // namespace XCorrAnalysis
