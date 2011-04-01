//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpEncoder...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpEncoder.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/encoder.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpEncoder)

namespace {
  
  // name of the logger to be used with MsgLogger
  const char* logger = "DumpEncoder"; 
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpEncoder::DumpEncoder (const std::string& name)
  : Module(name)
{
  m_src = configStr("encoderSource", "DetInfo(:Encoder)");
}

//--------------
// Destructor --
//--------------
DumpEncoder::~DumpEncoder ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpEncoder::beginCalibCycle(Env& env)
{
  MsgLog(logger, info, name() << ": in beginCalibCycle()");

  shared_ptr<Psana::Encoder::ConfigV1> config1 = env.configStore().get(m_src);
  if (config1.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "Encoder::ConfigV1:";
      str << "\n  chan_num = " << config1->chan_num();
      str << "\n  count_mode = " << config1->count_mode();
      str << "\n  quadrature_mode = " << config1->quadrature_mode();
      str << "\n  input_num = " << config1->input_num();
      str << "\n  input_rising = " << config1->input_rising();
      str << "\n  ticks_per_sec = " << config1->ticks_per_sec();
    }
    
  }
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpEncoder::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Encoder::DataV1> data1 = evt.get(m_src);
  if (data1.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "Encoder::DataV1:"
          << " timestamp = " << data1->timestamp()
          << " encoder_count = " << data1->encoder_count();
    }
  }

  shared_ptr<Psana::Encoder::DataV2> data2 = evt.get(m_src);
  if (data2.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "Encoder::DataV2:"
          << " timestamp = " << data2->timestamp()
          << " encoder_count =";
      const uint32_t* counts = data2->encoder_count();
      for (int i = 0; i != Psana::Encoder::DataV2::NEncoders; ++ i) {
        str << " " << counts[i];
      }

    }
  }

}
  
} // namespace psana_examples
