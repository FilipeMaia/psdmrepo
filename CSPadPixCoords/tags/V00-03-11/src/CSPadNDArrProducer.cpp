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
  , m_print_bits()
  , m_count(0)
{
  // get the values from configuration or use defaults
  m_source        = configSrc("source",        ":Cspad.0");
  m_inkey         = configStr("inkey",         "");
  m_outkey        = configStr("outkey",        "cspad_ndarr");
  m_outtype       = configStr("outtype",       "float");
  m_print_bits    = config   ("print_bits",    0);

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
        << "\nprint_bits    : "     << m_print_bits;
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

/// Method which is called at the beginning of the run
void 
CSPadNDArrProducer::beginRun(Event& evt, Env& env)
{
  // getQuadConfigPars(env); // DO NOT NEED THEM TO COPY ENTIRE ARRAY
}

//--------------------

/// Method which is called at the beginning of the calibration cycle
void 
CSPadNDArrProducer::beginCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
CSPadNDArrProducer::event(Event& evt, Env& env)
{
  ++m_count;

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
  
/// Method which is called at the end of the calibration cycle
void 
CSPadNDArrProducer::endCalibCycle(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called at the end of the run
void 
CSPadNDArrProducer::endRun(Event& evt, Env& env)
{
}

//--------------------

/// Method which is called once at the end of the job
void 
CSPadNDArrProducer::endJob(Event& evt, Env& env)
{
}

//--------------------

void 
CSPadNDArrProducer::getQuadConfigPars(Env& env)
{
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV2>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV3>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV4>(env) ) return;
  if ( getQuadConfigParsForType<Psana::CsPad::ConfigV5>(env) ) return;

  MsgLog(name(), warning, "CsPad::ConfigV2 - V5 is not available in this run.");
}

//--------------------

void 
CSPadNDArrProducer::getCSPadConfigFromData(Event& evt)
{
  if ( getCSPadConfigFromDataForType <Psana::CsPad::DataV1, Psana::CsPad::ElementV1> (evt) ) return;
  if ( getCSPadConfigFromDataForType <Psana::CsPad::DataV2, Psana::CsPad::ElementV2> (evt) ) return;

  MsgLog(name(), warning, "getCSPadConfigFromData(...): Psana::CsPad::DataV# / ElementV# for #=[2-5] is not available in this event.");
}

//--------------------

void 
CSPadNDArrProducer::procEvent(Event& evt, Env& env)
{  
  getCSPadConfigFromData(evt);

  // proc event  for one of the supported data types
  if ( m_outtype == "float"   and procEventForOutputType<float>   (evt) ) return; 
  if ( m_outtype == "double"  and procEventForOutputType<double>  (evt) ) return; 
  if ( m_outtype == "int"     and procEventForOutputType<int>     (evt) ) return; 
  if ( m_outtype == "int16"   and procEventForOutputType<int16_t> (evt) ) return; 
  if ( m_outtype == "int16_t" and procEventForOutputType<int16_t> (evt) ) return; 
}

//--------------------

void 
CSPadNDArrProducer::checkTypeImplementation()
{  
  if ( m_outtype == "float"   ) return; 
  if ( m_outtype == "double"  ) return; 
  if ( m_outtype == "int"     ) return; 
  if ( m_outtype == "int16"   ) return; 
  if ( m_outtype == "int16_t" ) return; 

  const std::string msg = "The requested data type: " + m_outtype + " is not implemented";
  MsgLog(name(), warning, msg );
  throw std::runtime_error(msg);
}

//--------------------

} // namespace CSPadPixCoords
