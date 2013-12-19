//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CSPad2x2NDArrProducer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "CSPadPixCoords/CSPad2x2NDArrProducer.h"

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
PSANA_MODULE_FACTORY(CSPad2x2NDArrProducer)

using namespace std;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace CSPadPixCoords {

//----------------
// Constructors --
//----------------

CSPad2x2NDArrProducer::CSPad2x2NDArrProducer (const std::string& name)
  : Module(name)
  , m_source()
  , m_inkey()
  , m_outkey()
  , m_outtype()
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_source        = configSrc("source",     ":Cspad2x2.0");
  m_inkey         = configStr("inkey",      "");
  m_outkey        = configStr("outkey",     "cspad2x2_ndarr");
  m_outtype       = configStr("outtype",    "float");
  m_print_bits    = config   ("print_bits", 0);

  checkTypeImplementation();
  m_config = new CONFIG ( m_source ); 
}

//--------------
// Destructor --
//--------------

CSPad2x2NDArrProducer::~CSPad2x2NDArrProducer ()
{
}

//--------------------

/// Print input parameters
void 
CSPad2x2NDArrProducer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\nInput parameters:"
        << "\nsource        : "     << m_source      
        << "\ninkey         : "     << m_inkey        
        << "\noutkey        : "     << m_outkey       
        << "\nouttype       : "     << m_outtype
        << "\ndtype         : "     << m_dtype
        << "\nprint_bits    : "     << m_print_bits
        << "\n";
  }
}

//--------------------

/// Method which is called once at the beginning of the job
void 
CSPad2x2NDArrProducer::beginJob(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
  if( m_print_bits & 16) printSizeOfTypes();
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPad2x2NDArrProducer::event(Event& evt, Env& env)
{
  ++m_count;

  if( ! m_config -> isSet() ) m_config -> setCSPad2x2ConfigPars(evt, env); 
  if( m_count==1 && m_print_bits & 2 ) m_config -> printCSPad2x2ConfigPars();

  struct timespec start, stop;
  int status = clock_gettime( CLOCK_REALTIME, &start ); // Get LOCAL time

  procEvent(evt, env);

  if( m_print_bits & 4 ) {
    status = clock_gettime( CLOCK_REALTIME, &stop ); // Get LOCAL time
    cout << "  Time to produce cspad2x2 ndarray is " 
         << stop.tv_sec - start.tv_sec + 1e-9*(stop.tv_nsec - start.tv_nsec) 
         << " sec" << endl;
  }
}

//--------------------

void CSPad2x2NDArrProducer::beginRun(Event& evt, Env& env) {}
void CSPad2x2NDArrProducer::beginCalibCycle(Event& evt, Env& env) {}
void CSPad2x2NDArrProducer::endCalibCycle(Event& evt, Env& env) {}
void CSPad2x2NDArrProducer::endRun(Event& evt, Env& env) {}
void CSPad2x2NDArrProducer::endJob(Event& evt, Env& env) {}

//--------------------

void 
CSPad2x2NDArrProducer::procEvent(Event& evt, Env& env)
{  
  // proc event  for one of the supported data types
  if ( m_dtype == FLOAT   and procEventForOutputType<float>   (evt) ) return; 
  if ( m_dtype == DOUBLE  and procEventForOutputType<double>  (evt) ) return; 
  if ( m_dtype == INT     and procEventForOutputType<int>     (evt) ) return; 
  if ( m_dtype == INT16   and procEventForOutputType<int16_t> (evt) ) return; 
}

//--------------------

void 
CSPad2x2NDArrProducer::checkTypeImplementation()
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
