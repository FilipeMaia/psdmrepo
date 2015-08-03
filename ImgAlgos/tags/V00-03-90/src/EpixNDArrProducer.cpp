//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixNDArrProducer...
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/EpixNDArrProducer.h"

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

PSANA_MODULE_FACTORY(EpixNDArrProducer)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
EpixNDArrProducer::EpixNDArrProducer (const std::string& name)
  : Module(name)
  , m_str_src()
  , m_key_in()
  , m_key_out()
  , m_outtype()
  , m_print_bits()
{
    m_str_src    = configSrc("source",     "DetInfo(:Epix)"); // :Epix10k or :Epix100a
    m_key_in     = configStr("key_in",     "");
    m_key_out    = configStr("key_out",    "epix-ndarr");
    m_outtype    = configStr("outtype",    "asdata");
    m_print_bits = config   ("print_bits", 0);

    checkTypeImplementation();
}

//--------------
// Destructor --
//--------------
EpixNDArrProducer::~EpixNDArrProducer ()
{
}

//--------------------
// Method which is called once at the beginning of the job
void 
EpixNDArrProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
  if( m_print_bits & 8 ) printSizeOfTypes();
}

//--------------------
// Method which is called at the beginning of the calibration cycle
void 
EpixNDArrProducer::beginCalibCycle(Event& evt, Env& env)
{
  MsgLog(name(), trace, "in beginCalibCycle()");
  if ( getConfigData<Psana::Epix::ConfigV1>    (env, "ConfigV1"   )  ) return;
  if ( getConfigData<Psana::Epix::Config10KV1> (env, "Config10KV1")  ) return;
  if ( getConfigData<Psana::Epix::Config100aV1>(env, "Config100aV1") ) return;
  //if ( getConfigData<Psana::Epix::Config100aV2>(env, "Config100aV2") ) return;
}

//--------------------
// Method which is called with event data
void 
EpixNDArrProducer::event(Event& evt, Env& env)
{
  procEvent(evt, env);
}

//--------------------

void 
EpixNDArrProducer::procEvent(Event& evt, Env& env)
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
EpixNDArrProducer::checkTypeImplementation()
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
EpixNDArrProducer::printInputParameters()
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
