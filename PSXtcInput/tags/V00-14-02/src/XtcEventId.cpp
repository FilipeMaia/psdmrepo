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
XtcEventId::XtcEventId (int run, const PSTime::Time& time, unsigned fiducials, unsigned ticks, unsigned vector, unsigned control)
  : PSEvt::EventId()
  , m_run(run)
  , m_time(time)
  , m_fiducials(fiducials)
  , m_ticks(ticks)
  , m_vector(vector)
  , m_control(control)
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

/**
 *  @brief Returns fiducials counter for the event.
 */
unsigned
XtcEventId::fiducials() const
{
  return m_fiducials;
}

/**
 *  @brief Returns 119MHz counter within the fiducial.
 */
unsigned
XtcEventId::ticks() const
{
  return m_ticks;
}

/**
 *  @brief Returns event counter since Configure.
 */
unsigned
XtcEventId::vector() const
{
  return m_vector;
}

/**
 *  @brief Returns control - alternate representation of Xtc header.
 */
unsigned
XtcEventId::control() const
{
  return m_control;
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
  os << "XtcEventId(run=" << m_run << ", time=" << m_time << ", fiducials=" << m_fiducials
     << ", ticks=" << m_ticks << ", vector=" << m_vector << ", control=" << m_control << ")";
}

} // namespace PSXtcInput
