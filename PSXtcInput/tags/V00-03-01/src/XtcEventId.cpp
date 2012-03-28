//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcEventId...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcInput/XtcEventId.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace PSXtcInput {

//----------------
// Constructors --
//----------------
XtcEventId::XtcEventId (int run, const PSTime::Time& time)
  : PSEvt::EventId()
  , m_run(run)
  , m_time(time)
{
}

//--------------
// Destructor --
//--------------
XtcEventId::~XtcEventId ()
{
}

/**
 *  @brief Return the time for event.
 */
PSTime::Time 
XtcEventId::time() const
{
  return m_time;
}

/**
 *  @brief Return the run number for event.
 *  
 *  If run number is not known -1 will be returned.
 */
int 
XtcEventId::run() const
{
  return m_run;
}

/// check if two event IDs refer to the same event
bool 
XtcEventId::operator==(const EventId& other) const
{
  return m_time == other.time();
}

/// Compare two event IDs for ordering purpose
bool 
XtcEventId::operator<(const EventId& other) const
{
  return m_time < other.time();
}

/// Dump object in human-readable format
void 
XtcEventId::print(std::ostream& os) const
{
  os << "XtcEventId(run=" << m_run << ", time=" << m_time << ')'; 
}

} // namespace PSXtcInput
