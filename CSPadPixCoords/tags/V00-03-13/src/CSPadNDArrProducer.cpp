//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPadNDArrProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/CSPadNDArrProducer.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <time.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
//#include <boost/lexical_cast.hpp>

// This declares this class as psana module
using namespace CSPadPixCoords;
PSANA_MODULE_FACTORY(CSPadNDArrProducer)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

CSPadNDArrProducer::CSPadNDArrProducer (const std::string& name)
  : Module(name)
  , m_source()
  , m_inkey()
  , m_outkey()
  , m_outtype()
  , m_is_fullsize()  
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_source        = configSrc("source",       ":Cspad.0");
  m_inkey         = configStr("inkey",        "");
  m_outkey        = configStr("outkey",       "cspad_ndarr");
  m_outtype       = configStr("outtype",      "float");
  m_is_fullsize   = config   ("is_fullsize",  false);
  m_print_bits    = config   ("print_bits",   0);

  checkTypeImplementation();
}

//--------------
// Destructor --
//--------------

CSPadNDArrProducer::~CSPadNDArrProducer ()
{
}

//--------------------

/// Print input parameters
void 
CSPadNDArrProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource        : "     << m_source      
        << "\ninkey         : "     << m_inkey        
        << "\noutkey        : "     << m_outkey       
        << "\nouttype       : "     << m_outtype
        << "\ndtype         : "     << m_dtype
        << "\nis_fullsize   : "     << m_is_fullsize
        << "\nprint_bits    : "     << m_print_bits
        << "\n";
  }
}

//--------------------

/// Method which is called once at the beginning of the job
void 
CSPadNDArrProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadNDArrProducer::event(Event& evt, Env& env)
{
  ++m_count;

  if (m_count==1) getConfigParameters(evt, env);

  struct timespec start, stop;
  int status = clock_gettime( CLOCK_REALTIME, &start ); // Get LOCAL time

  procEvent(evt, env);

  if( m_print_bits & 4 ) {
    status = clock_gettime( CLOCK_REALTIME, &stop ); // Get LOCAL time
    cout << "  Time to produce cspad ndarray is " 
         << stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec - start.tv_nsec) 
         << " sec" << endl;
  }
}

//--------------------

void CSPadNDArrProducer::getConfigParameters(Event& evt, Env& env)
{
  m_config = new CONFIG ( m_source );
  m_config -> setCSPadConfigPars (evt, env);
  if( m_print_bits & 2 ) m_config -> printCSPadConfigPars();
}

//--------------------

void CSPadNDArrProducer::beginRun(Event& evt, Env& env) {}
void CSPadNDArrProducer::beginCalibCycle(Event& evt, Env& env) {}
void CSPadNDArrProducer::endCalibCycle(Event& evt, Env& env) {}
void CSPadNDArrProducer::endRun(Event& evt, Env& env) {}
void CSPadNDArrProducer::endJob(Event& evt, Env& env) {}
  
//--------------------

void 
CSPadNDArrProducer::procEvent(Event& evt, Env& env)
{  
  // proc event  for one of the supported data types
  if ( m_dtype == FLOAT   and procEventForOutputType<float>   (evt) ) return; 
  if ( m_dtype == DOUBLE  and procEventForOutputType<double>  (evt) ) return; 
  if ( m_dtype == INT     and procEventForOutputType<int>     (evt) ) return; 
  if ( m_dtype == INT16   and procEventForOutputType<int16_t> (evt) ) return; 
}

//--------------------

void 
CSPadNDArrProducer::checkTypeImplementation()
{  
  if ( m_outtype == "float"   ) { m_dtype = FLOAT;  return; }
  if ( m_outtype == "double"  ) { m_dtype = DOUBLE; return; } 
  if ( m_outtype == "int"     ) { m_dtype = INT;    return; } 
  if ( m_outtype == "int16"   ) { m_dtype = INT16;  return; } 
  if ( m_outtype == "int16_t" ) { m_dtype = INT16;  return; } 

  const std::string msg = "The requested data type: " + m_outtype + " is not implemented";
  MsgLog(name(), warning, msg );
  throw std::runtime_error(msg);
}

//--------------------

} // namespace CSPadPixCoords
