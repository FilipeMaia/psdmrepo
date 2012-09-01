//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PnccdImageProducer.cpp 0001 2012-07-06 09:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class PnccdImageProducer...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/PnccdImageProducer.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "psddl_psana/pnccd.ddl.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(PnccdImageProducer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
PnccdImageProducer::PnccdImageProducer (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out()
  , m_print_bits()
 {
    m_str_src    = configStr("source",    "DetInfo(:pnCCD)");
    m_key_in     = configStr("inkey",     "");
    m_key_out    = configStr("outimgkey", "pnccdimg");
    m_print_bits = config   ("print_bits", 0);
}

//--------------
// Destructor --
//--------------
PnccdImageProducer::~PnccdImageProducer ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
PnccdImageProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

// Method which is called at the beginning of the calibration cycle
void 
PnccdImageProducer::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");

  shared_ptr<Psana::PNCCD::ConfigV1> config1 = env.configStore().get(m_str_src);
  if (config1.get()) {
    
    WithMsgLog(name(), info, str) {
      str << "PNCCD::ConfigV1:";
      str << "\n  numLinks = " << config1->numLinks();
      str << "\n  payloadSizePerLink = " << config1->payloadSizePerLink();
    }    
  }

  shared_ptr<Psana::PNCCD::ConfigV2> config2 = env.configStore().get(m_str_src);
  if (config2.get()) {    
    if( m_print_bits & 4 ) {
      WithMsgLog(name(), info, str) {
        str << "PNCCD::ConfigV2:";
        str << "\n  numLinks = "             << config2->numLinks();
        str << "\n  payloadSizePerLink = "   << config2->payloadSizePerLink();
        str << "\n  numChannels = "          << config2->numChannels();
        str << "\n  numRows = "              << config2->numRows();
        str << "\n  numSubmoduleChannels = " << config2->numSubmoduleChannels();
        str << "\n  numSubmoduleRows = "     << config2->numSubmoduleRows();
        str << "\n  numSubmodules = "        << config2->numSubmodules();
        str << "\n  camexMagic = "           << config2->camexMagic();
        str << "\n  info = "                 << config2->info();
        str << "\n  timingFName = "          << config2->timingFName();
      } 
    }
  }
}

//--------------------
// Method which is called with event data
void 
PnccdImageProducer::event(Event& evt, Env& env)
{
  shared_ptr<Psana::PNCCD::FullFrameV1> frame = evt.get(m_str_src, m_key_in, &m_src);
  if (frame) {

      const ndarray<uint16_t, 2> data = frame->data();

      if( m_print_bits & 2 ) {
        for (int i=0; i<10; ++i) cout << " " << data[0][i];
        cout << "\n";
      }

      save2DArrayInEvent<uint16_t> (evt, m_src, m_key_out, data.data(), data.shape());
  }
  else
  {
      const std::string msg = "PNCCD::FullFrameV1 object is not available in the event(...) for source:" 
                            + m_str_src + " key:" + m_key_in;
      MsgLog(name(), warning, msg);       
  }
}

//--------------------
/// Print input parameters
void 
PnccdImageProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource       : "     << m_str_src
        << "\ninkey        : "     << m_key_in      
        << "\noutimgkey    : "     << m_key_out
        << "\nm_print_bits : "     << m_print_bits;
  }
}

//--------------------


//--------------------

} // namespace ImgAlgos
