//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DumpLusi...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_examples/DumpLusi.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/lusi.ddl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_examples;
PSANA_MODULE_FACTORY(DumpLusi)

namespace {
  
  // name of the logger to be used with MsgLogger
  const char* logger = "DumpLusi"; 
  
}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace psana_examples {

//----------------
// Constructors --
//----------------
DumpLusi::DumpLusi (const std::string& name)
  : Module(name)
{
  m_ipimbSrc = configStr("ipimbSource", "DetInfo(:Ipimb)");
  m_tmSrc = configStr("tmSource", "DetInfo(:Tm6740)");
}

//--------------
// Destructor --
//--------------
DumpLusi::~DumpLusi ()
{
}

/// Method which is called at the beginning of the calibration cycle
void 
DumpLusi::beginCalibCycle(Env& env)
{
  MsgLog(logger, info, name() << ": in beginCalibCycle()");

  shared_ptr<Psana::Lusi::DiodeFexConfigV1> dconfig = env.configStore().get(m_ipimbSrc);
  if (dconfig.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "Lusi::DiodeFexConfigV1:";
      const float* base = dconfig->base();
      const float* scale = dconfig->scale();
      str << "\n  base =";
      for (int i = 0; i < Psana::Lusi::DiodeFexConfigV1::NRANGES; ++ i) {
        str << " " << base[i];
      }
      str << "\n  scale =";
      for (int i = 0; i < Psana::Lusi::DiodeFexConfigV1::NRANGES; ++ i) {
        str << " " << scale[i];
      }
    }
    
  }

  shared_ptr<Psana::Lusi::IpmFexConfigV1> iconfig = env.configStore().get(m_ipimbSrc);
  if (iconfig.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "Psana::Lusi::IpmFexConfigV1:";
      str << "\n  xscale = " << iconfig->xscale();
      str << "\n  yscale = " << iconfig->yscale();
      for (int ch = 0; ch < Psana::Lusi::IpmFexConfigV1::NCHANNELS; ++ ch) {
        str << "\n  channel #" << ch << ":";
        
        const Psana::Lusi::DiodeFexConfigV1& dconfig = iconfig->diode(ch);
        const float* base = dconfig.base();
        const float* scale = dconfig.scale();
        str << "\n    base =";
        for (int i = 0; i < Psana::Lusi::DiodeFexConfigV1::NRANGES; ++ i) {
          str << " " << base[i];
        }
        str << "\n    scale =";
        for (int i = 0; i < Psana::Lusi::DiodeFexConfigV1::NRANGES; ++ i) {
          str << " " << scale[i];
        }
      }
    }
    
  }

  shared_ptr<Psana::Lusi::PimImageConfigV1> pconfig = env.configStore().get(m_tmSrc);
  if (pconfig.get()) {
    
    WithMsgLog(logger, info, str) {
      str << "Psana::Lusi::PimImageConfigV1:";
      str << "\n  xscale = " << iconfig->xscale();
      str << "\n  yscale = " << iconfig->yscale();
    }
    
  }

}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
DumpLusi::event(Event& evt, Env& env)
{
  shared_ptr<Psana::Lusi::DiodeFexV1> diode = evt.get(m_ipimbSrc);
  if (diode.get()) {
    WithMsgLog(logger, info, str) {
      str << "Lusi::DiodeFexV1: value = " << diode->value();
    }
  }

  shared_ptr<Psana::Lusi::IpmFexV1> fex = evt.get(m_ipimbSrc);
  if (fex.get()) {
    WithMsgLog(logger, info, str) {
      str << "Psana::Lusi::IpmFexV1:";
      str << "\n  sum = " << fex->sum();
      str << "\n  xpos = " << fex->xpos();
      str << "\n  ypos = " << fex->ypos();

      const float* channel = fex->channel();
      str << "\n  channel =";
      for (int i = 0; i < Psana::Lusi::IpmFexV1::NCHANNELS; ++ i) {
        str << " " << channel[i];
      }
    }
  }

}

} // namespace psana_examples
