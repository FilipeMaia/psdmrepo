//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpEvr...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpEvr.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/evr.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpEvr)

namespace {
  
  // name of the logger to be used with MsgLogger
  const char* logger = "DumpEvr"; 
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpEvr::DumpEvr (const std::string& name)
  : Module(name)
{
  m_src = configStr("encoderSource", "DetInfo(:Evr)");
}

//--------------
// Destructor --
//--------------
DumpEvr::~DumpEvr ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpEvr::beginCalibCycle(Env& env)
{
  MsgLog(logger, info, name() << ": in beginCalibCycle()");

  // Try to get V1 config object
  shared_ptr<Psana::EvrData::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "EvrData::ConfigV1: npulses = " << config1->npulses()
          << " noutputs = " << config1->noutputs();
      for (unsigned i = 0; i < config1->npulses(); ++ i) {
        const Psana::EvrData::PulseConfig& pcfg = config1->pulses(i);
        str << "\n  pulse config #" << i 
            << ": pulse=" << pcfg.pulse()
            << " polarity=" << int(pcfg.polarity())
            << " prescale=" << pcfg.prescale()
            << " delay=" << pcfg.delay()
            << " width=" << pcfg.width();
      }
      for (unsigned i = 0; i < config1->noutputs(); ++ i) {
        const Psana::EvrData::OutputMap& ocfg = config1->output_maps(i);
        str << "\n  output config #" << i 
            << ": source=" << ocfg.source()
            << " source_id=" << int(ocfg.source_id())
            << " conn=" << ocfg.conn()
            << " conn_id=" << int(ocfg.conn_id());
      }
    }
    
  }

  // Try to get V2 config object
  shared_ptr<Psana::EvrData::ConfigV2> config2 = env.configStore().get(m_src);
  if (config2.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "EvrData::ConfigV2: npulses = " << config2->npulses()
          << " noutputs = " << config2->noutputs()
          << " beam = " << config2->beam()
          << " rate = " << config2->rate() ;
      for (unsigned i = 0; i < config2->npulses(); ++ i) {
        const Psana::EvrData::PulseConfig& pcfg = config2->pulses(i);
        str << "\n  pulse config #" << i 
            << ": pulse=" << pcfg.pulse()
            << " polarity=" << int(pcfg.polarity())
            << " prescale=" << pcfg.prescale()
            << " delay=" << pcfg.delay()
            << " width=" << pcfg.width();
      }
      for (unsigned i = 0; i < config2->noutputs(); ++ i) {
        const Psana::EvrData::OutputMap& ocfg = config2->output_maps(i);
        str << "\n  output config #" << i 
            << ": source=" << ocfg.source()
            << " source_id=" << int(ocfg.source_id())
            << " conn=" << ocfg.conn()
            << " conn_id=" << int(ocfg.conn_id());
      }
    }
    
  }

  // Try to get V3 config object
  shared_ptr<Psana::EvrData::ConfigV3> config3 = env.configStore().get(m_src);
  if (config3.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "EvrData::ConfigV3: npulses = " << config3->npulses()
          << " noutputs = " << config3->noutputs()
          << " neventcodes = " << config3->neventcodes();
      for (unsigned i = 0; i < config3->npulses(); ++ i) {
        const Psana::EvrData::PulseConfigV3& pcfg = config3->pulses(i);
        str << "\n  pulse config #" << i 
            << ": pulseId=" << pcfg.pulseId()
            << " polarity=" << int(pcfg.polarity())
            << " prescale=" << pcfg.prescale()
            << " delay=" << pcfg.delay()
            << " width=" << pcfg.width();
      }
      for (unsigned i = 0; i < config3->noutputs(); ++ i) {
        const Psana::EvrData::OutputMap& ocfg = config3->output_maps(i);
        str << "\n  output config #" << i 
            << ": source=" << ocfg.source()
            << " source_id=" << int(ocfg.source_id())
            << " conn=" << ocfg.conn()
            << " conn_id=" << int(ocfg.conn_id());
      }
      for (unsigned i = 0; i < config3->neventcodes(); ++ i) {
        const Psana::EvrData::EventCodeV3& ecfg = config3->eventcodes(i);
        str << "\n  event code #" << i 
            << ": code=" << ecfg.code()
            << " isReadout=" << int(ecfg.isReadout())
            << " isTerminator=" << int(ecfg.isTerminator())
            << " maskTrigger=" << ecfg.maskTrigger()
            << " maskSet=" << ecfg.maskSet()
            << " maskClear=" << ecfg.maskClear();
      }
    }
    
  }

  // Try to get V4 config object
  shared_ptr<Psana::EvrData::ConfigV4> config4 = env.configStore().get(m_src);
  if (config4.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "EvrData::ConfigV4: npulses = " << config4->npulses()
          << " noutputs = " << config4->noutputs()
          << " neventcodes = " << config4->neventcodes();
      for (unsigned i = 0; i < config4->npulses(); ++ i) {
        const Psana::EvrData::PulseConfigV3& pcfg = config4->pulses(i);
        str << "\n  pulse config #" << i 
            << ": pulseId=" << pcfg.pulseId()
            << " polarity=" << int(pcfg.polarity())
            << " prescale=" << pcfg.prescale()
            << " delay=" << pcfg.delay()
            << " width=" << pcfg.width();
      }
      for (unsigned i = 0; i < config4->noutputs(); ++ i) {
        const Psana::EvrData::OutputMap& ocfg = config4->output_maps(i);
        str << "\n  output config #" << i 
            << ": source=" << ocfg.source()
            << " source_id=" << int(ocfg.source_id())
            << " conn=" << ocfg.conn()
            << " conn_id=" << int(ocfg.conn_id());
      }
      for (unsigned i = 0; i < config4->neventcodes(); ++ i) {
        const Psana::EvrData::EventCodeV4& ecfg = config4->eventcodes(i);
        str << "\n  event code #" << i 
            << ": code=" << ecfg.code()
            << " isReadout=" << int(ecfg.isReadout())
            << " isTerminator=" << int(ecfg.isTerminator())
            << " reportDelay=" << ecfg.reportDelay()
            << " reportWidth=" << ecfg.reportWidth()
            << " maskTrigger=" << ecfg.maskTrigger()
            << " maskSet=" << ecfg.maskSet()
            << " maskClear=" << ecfg.maskClear();
      }
    }
    
  }

  // Try to get V5 config object
  shared_ptr<Psana::EvrData::ConfigV5> config5 = env.configStore().get(m_src);
  if (config5.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "EvrData::ConfigV5: npulses = " << config5->npulses()
          << " noutputs = " << config5->noutputs()
          << " neventcodes = " << config5->neventcodes();
      for (unsigned i = 0; i < config5->npulses(); ++ i) {
        const Psana::EvrData::PulseConfigV3& pcfg = config5->pulses(i);
        str << "\n  pulse config #" << i 
            << ": pulseId=" << pcfg.pulseId()
            << " polarity=" << int(pcfg.polarity())
            << " prescale=" << pcfg.prescale()
            << " delay=" << pcfg.delay()
            << " width=" << pcfg.width();
      }
      for (unsigned i = 0; i < config5->noutputs(); ++ i) {
        const Psana::EvrData::OutputMap& ocfg = config5->output_maps(i);
        str << "\n  output config #" << i 
            << ": source=" << ocfg.source()
            << " source_id=" << int(ocfg.source_id())
            << " conn=" << ocfg.conn()
            << " conn_id=" << int(ocfg.conn_id());
      }
      for (unsigned i = 0; i < config5->neventcodes(); ++ i) {
        const Psana::EvrData::EventCodeV5& ecfg = config5->eventcodes(i);
        str << "\n  event code #" << i 
            << ": code=" << ecfg.code()
            << " isReadout=" << int(ecfg.isReadout())
            << " isTerminator=" << int(ecfg.isTerminator())
            << " isLatch=" << int(ecfg.isLatch())
            << " reportDelay=" << ecfg.reportDelay()
            << " reportWidth=" << ecfg.reportWidth()
            << " maskTrigger=" << ecfg.maskTrigger()
            << " maskSet=" << ecfg.maskSet()
            << " maskClear=" << ecfg.maskClear();
      }
      const Psana::EvrData::SequencerConfigV1& scfg = config5->seq_config();
      str << "\n  seq_config: sync_source=" << scfg.sync_source()
          << " beam_source=" << scfg.beam_source()
          << " length=" << scfg.length()
          << " cycles=" << scfg.cycles();
      for (unsigned i = 0; i < scfg.length(); ++ i) {
        const Psana::EvrData::SequencerEntry& e = scfg.entries(i);
        str << "\n    entry #" << i <<  " delay=" << e.delay() << " eventcode=" << e.eventcode();
      }
    }
    
  }

  shared_ptr<Psana::EvrData::IOConfigV1> iocfg1 = env.configStore().get(m_src);
  if (iocfg1.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "EvrData::IOConfigV1: nchannels = " << iocfg1->nchannels()
          << " conn = " << iocfg1->conn();
      for (unsigned i = 0; i < iocfg1->nchannels(); ++ i) {
        const Psana::EvrData::IOChannel& ioch = iocfg1->channels(i);
        str << "\n  io channel #" << i 
            << ": name=" << ioch.name()
            << " infos=[";
        for (unsigned d = 0; d != ioch.ninfo(); ++ d) {
          const Pds::DetInfo& dinfo = ioch.infos(d);
          str << " " << Pds::DetInfo::name(dinfo);
        }
        str << " ]";
      }
    }
  }
  
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpEvr::event(Event& evt, Env& env)
{
  shared_ptr<Psana::EvrData::DataV3> data3 = evt.get(m_src);
  if (data3.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "Encoder::DataV3: numFifoEvents=" << data3->numFifoEvents();
      for (unsigned i = 0; i < data3->numFifoEvents(); ++ i) {
        const Psana::EvrData::FIFOEvent& f = data3->fifoEvents(i);
        str << "\n    fifo event #" << i 
            <<  " timestampHigh=" << f.timestampHigh() 
            <<  " timestampLow=" << f.timestampLow() 
            << " eventCode=" << f.eventCode();
      }
    }
  }

}
  
} // namespace psana_examples
