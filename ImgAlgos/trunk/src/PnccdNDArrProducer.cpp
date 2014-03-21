//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: PnccdNDArrProducer.cpp 0001 2014-01-17 09:00:00Z dubrovin@SLAC.STANFORD.EDU $
//
// Description:
//	Class PnccdNDArrProducer...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/PnccdNDArrProducer.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "ImgAlgos/GlobalMethods.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
using namespace Psana;
using namespace ImgAlgos;

PSANA_MODULE_FACTORY(PnccdNDArrProducer)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
PnccdNDArrProducer::PnccdNDArrProducer (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out()
  , m_outtype()
  , m_print_bits()
{
    m_str_src    = configSrc("source",     "DetInfo(:pnCCD)");  // DetInfo(Camp.0:pnCCD.1)
    m_key_in     = configStr("key_in",     "");
    m_key_out    = configStr("key_out",    "pnccd-ndarr");
    m_outtype    = configStr("outtype",    "asdata");
    m_print_bits = config   ("print_bits", 0);

    checkTypeImplementation();
}

//--------------
// Destructor --
//--------------
PnccdNDArrProducer::~PnccdNDArrProducer ()
{
}

//--------------------

/// Method which is called once at the beginning of the job
void 
PnccdNDArrProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
  if( m_print_bits & 8 ) printSizeOfTypes();
}

// Method which is called at the beginning of the calibration cycle
void 
PnccdNDArrProducer::beginCalibCycle(Event& evt, Env& env)
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
PnccdNDArrProducer::event(Event& evt, Env& env)
{
  procEvent(evt, env);
}

//--------------------

void 
PnccdNDArrProducer::procEvent(Event& evt, Env& env)
{  
  // proc event  for one of the supported data types
  if ( m_dtype == ASDATA  and procEventForOutputType<data_t>   (evt) ) return;
  if ( m_dtype == FLOAT   and procEventForOutputType<float>    (evt) ) return;
  if ( m_dtype == DOUBLE  and procEventForOutputType<double>   (evt) ) return;
  if ( m_dtype == INT     and procEventForOutputType<int>      (evt) ) return;
  if ( m_dtype == INT16   and procEventForOutputType<int16_t>  (evt) ) return;
}

//--------------------

void 
PnccdNDArrProducer::checkTypeImplementation()
{  
  if ( m_outtype == "asdata"  ) { m_dtype = ASDATA; return; }
  if ( m_outtype == "float"   ) { m_dtype = FLOAT;  return; }
  if ( m_outtype == "double"  ) { m_dtype = DOUBLE; return; } 
  if ( m_outtype == "int"     ) { m_dtype = INT;    return; } 
  if ( m_outtype == "int16"   ) { m_dtype = INT16;  return; } 

  const std::string msg = "The requested data type: " + m_outtype + " is not implemented";
  MsgLog(name(), warning, msg );
  throw std::runtime_error(msg);
}

//--------------------
/// Print input parameters
void 
PnccdNDArrProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
        << "\n source     : "     << m_str_src
        << "\n key_in     : "     << m_key_in      
        << "\n key_out    : "     << m_key_out
        << "\n outtype    : "     << m_outtype
        << "\n print_bits : "     << m_print_bits
        << "\n";
  }
}

//--------------------

} // namespace ImgAlgos
