//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Tahometer...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/Tahometer.h"

//-----------------
// C/C++ Headers --
//-----------------
#define _USE_MATH_DEFINES // for M_PI
#include <cmath> // for sqrt, atan2
//#include <math.h> // for exp, M_PI
#include <fstream> // for ofstream

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"
#include "ImgAlgos/GlobalMethods.h"
//#include "PSTime/Time.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
#include <iomanip> // for setw, setfill
#include <sstream> // for stringstream
#include <iostream>// for setf

// This declares this class as psana module
using namespace std;
using namespace ImgAlgos;
PSANA_MODULE_FACTORY(Tahometer)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
Tahometer::Tahometer (const std::string& name)
  : Module(name)
  , m_print_bits()
  , m_dn()
  , m_count_dn(0)
  , m_count(0)
{
  m_dn         = config   ("dn",       100);
  m_print_bits = config   ("print_bits", 2);
}

//--------------
// Destructor --
//--------------
Tahometer::~Tahometer ()
{
}

/// Method which is called once at the beginning of the job
void 
Tahometer::beginJob(Event& evt, Env& env)
{
  m_time    = new TimeInterval();
  m_time_dn = new TimeInterval();
}

/// Method which is called at the beginning of the run
void 
Tahometer::beginRun(Event& evt, Env& env)
{
  if( m_print_bits & 1 ) printInputParameters();
  m_str_runnum = stringRunNumber(evt);
}

/// Method which is called at the beginning of the calibration cycle
void 
Tahometer::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
Tahometer::event(Event& evt, Env& env)
{
  procEvent(evt,env);
}
  
/// Method which is called at the end of the calibration cycle
void 
Tahometer::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
Tahometer::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
Tahometer::endJob(Event& evt, Env& env)
{
  double dt_sec = m_time -> getCurrentTimeInterval();

  if( m_print_bits & 2 ) {
    MsgLog(name(), info, "===== Summary for " << m_count << " processed events =====");
    //m_time -> stopTime(m_count);
    printTimeIntervalSummary(evt, dt_sec, m_count);
  }

  if( m_print_bits & 8 ) printSummaryForParser(evt, dt_sec, m_count);
}

//--------------------
//--------------------
//--------------------
// Print input parameters
void 
Tahometer::procEvent(Event& evt, Env& env)
{
  if(m_count < 1) {
    m_time    -> startTime();
    m_time_dn -> startTime();
    if( m_print_bits & 2 ) MsgLog(name(), info, "===== Start Tahometer: " << m_time -> strStartTime().c_str() << " =====");
  }

  ++ m_count;

  if( !(m_print_bits & 4)) return;
  if (++m_count_dn < m_dn) return;

  //m_time_dn -> stopTime(m_count_dn, false);
  double dt_sec = m_time_dn -> getCurrentTimeInterval();
  printTimeIntervalSummary(evt, dt_sec, m_count_dn);

  m_time_dn -> startTime();
  m_count_dn = 0;
}

//--------------------
// Print input parameters
void 
Tahometer::printInputParameters()
{
  WithMsgLog(name(), info, log) {
    log << "\n Input parameters:"
	<< "\n dn         : " << m_dn
	<< "\n print_bits : " << m_print_bits
        << "\n";     
  }
}

//--------------------
void 
Tahometer::printTimeIntervalSummary(Event& evt, double dt_sec, long counter)
{
  MsgLog( name(), info,  "Run="             << m_str_runnum 
                     << " Evt="             << stringFromUint(m_count) 
                     << " Time to process " << stringFromUint(counter) 
                     << " events is "       << dt_sec 
                     << " sec, or "         << dt_sec/counter << " sec/event" 
	           //<< " Time="   << stringTimeStamp(evt)
	           //<< comment.c_str() 
  );
}

//--------------------

void 
Tahometer::printSummaryForParser(Event& evt, double dt_sec, long counter)
{
  cout << "Tahometer: Summary for parser" << endl;
  cout << "BATCH_PROCESSING_TIME  " << dt_sec  << endl;
  cout << "BATCH_NUMBER_OF_EVENTS " << counter << endl;
  if (counter>0)
    cout << "BATCH_SEC_PER_EVENT    " << dt_sec/counter << endl;
  if (dt_sec>0)
    cout << "BATCH_EVENTS_PER_SEC   " << counter/dt_sec << endl;
}

//--------------------
//--------------------
//--------------------

} // namespace ImgAlgos

//---------EOF--------
