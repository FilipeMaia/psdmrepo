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

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(PnccdImageProducer)

using namespace std;

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
  , m_gap_rows()
  , m_gap_cols()
  , m_gap_value()
  , m_print_bits()
 {
    m_str_src    = configSrc("source",    "DetInfo(:pnCCD)");
    m_key_in     = configStr("inkey",     "");
    m_key_out    = configStr("outimgkey", "pnccdimg");
    m_gap_rows   = config   ("gap_rows",   0);
    m_gap_cols   = config   ("gap_cols",   0);
    m_gap_value  = config   ("gap_value",  0);
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
    if( m_print_bits & 4 ) {    
      WithMsgLog(name(), info, str) {
        str << "PNCCD::ConfigV1:";
        str << "\n  numLinks = " << config1->numLinks();
        str << "\n  payloadSizePerLink = " << config1->payloadSizePerLink();
      }
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
  if ( procEventForFullFrame<Psana::PNCCD::FullFrameV1> (evt) ) return;

  if ( procEventFor3DArrType<uint16_t> (evt) ) return; // as data
  if ( procEventFor3DArrType<float>    (evt) ) return;
  if ( procEventFor3DArrType<double>   (evt) ) return;
  if ( procEventFor3DArrType<int>      (evt) ) return;
  if ( procEventFor3DArrType<int16_t>  (evt) ) return;

  if( m_print_bits & 16 ) MsgLog(name(), warning, "PNCCD::FullFrameV1 or ndarray<T,3> object is not available in the event(...) for source:"
          << m_str_src << " key:" << m_key_in);
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
        << "\ngap_rows     : "     << m_gap_rows
        << "\ngap_cols     : "     << m_gap_cols
        << "\ngap_value    : "     << m_gap_value
        << "\nm_print_bits : "     << m_print_bits
        << "\n";
  }
}

//--------------------


//--------------------

} // namespace ImgAlgos
