//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeInterval...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ImgAlgos/TimeInterval.h"

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

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ImgAlgos {

//----------------
// Constructors --
//----------------
TimeInterval::TimeInterval()
{
  m_entrance_counter = 0;
  startTime();
}

//--------------
// Destructor --
//--------------
TimeInterval::~TimeInterval()
{
}

//--------------------
/// Store and prints time at start of the measured interval
void 
TimeInterval::startTimeOnce()
{
  if( m_entrance_counter > 0 ) return;
      m_entrance_counter ++;

  startTime();
}

//--------------------
/// Store and prints time at start of the measured interval
void 
TimeInterval::startTime()
{
  m_status = clock_gettime( CLOCK_REALTIME, &m_start ); // Get LOCAL time
}

//--------------------
/// Stop time interval
void 
TimeInterval::stopTime()
{
  m_status = clock_gettime( CLOCK_REALTIME, &m_stop ); // Get LOCAL time
}

//--------------------
/// Stop and prints time interval since start
void 
TimeInterval::stopTime(long nevents, bool print_at_stop)
{
  double dt = getCurrentTimeInterval();
  if(print_at_stop) MsgLog("TimeInterval::stopTime", info, "Time to process "<< nevents << " events is " << dt << " sec, or " << dt/nevents << " sec/event");
}

//--------------------
/// Get current time interval since start
double
TimeInterval::getCurrentTimeInterval()
{
  m_status = clock_gettime( CLOCK_REALTIME, &m_stop ); // Get LOCAL time
  double dt = m_stop.tv_sec - m_start.tv_sec + 1e-9*(m_stop.tv_nsec - m_start.tv_nsec);
  return dt;
}

//--------------------
/// Prints time at start of the measured interval
void 
TimeInterval::printStartTime()
{
  MsgLog("TimeInterval::startTime", info, "Start time: " << strStartTime().c_str() << " and " << m_start.tv_nsec << " nsec");
}

//--------------------
/// Prints time at stop of the measured interval
void 
TimeInterval::printStopTime()
{
  MsgLog("TimeInterval::stopTime", info, "Stop time: " << strStopTime().c_str() << " and " << m_stop.tv_nsec << " nsec");
}


//--------------------
/// Returns formatted string of the start time
std::string
TimeInterval::strStartTime(std::string fmt)
{
  return strTime(&m_start.tv_sec, fmt);
}

//--------------------
/// Returns formatted string of the stop time
std::string
TimeInterval::strStopTime(std::string fmt)
{
  return strTime(&m_stop.tv_sec, fmt);
}

//--------------------
/// Returns formatted string of the raw time in sec
std::string
TimeInterval::strTime(time_t* p_tsec, std::string fmt)
{
  struct tm * timeinfo; timeinfo = localtime ( p_tsec ); 
  char c_time_buf[80]; strftime (c_time_buf, 80, fmt.c_str(), timeinfo);
  return std::string(c_time_buf);
}

//--------------------

} // namespace ImgAlgos
