//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5EventId...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5EventId.h"

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

namespace PSHdf5Input {

//--------------
// Destructor --
//--------------
Hdf5EventId::~Hdf5EventId ()
{
}

/**
 *  @brief Return the time for event.
 */
PSTime::Time
Hdf5EventId::time() const
{
  return m_time;
}

/**
 *  @brief Return the run number for event.
 *
 *  If run number is not known -1 will be returned.
 */
int
Hdf5EventId::run() const
{
  return m_run;
}

// Returns fiducials counter for the event.
unsigned
Hdf5EventId::fiducials() const
{
  return 0;
}

// Returns event counter since Configure.
unsigned
Hdf5EventId::vector() const
{
  return 0;
}

/// check if two event IDs refer to the same event
bool
Hdf5EventId::operator==(const EventId& other) const
{
  return m_time == other.time();
}

/// Compare two event IDs for ordering purpose
bool
Hdf5EventId::operator<(const EventId& other) const
{
  return m_time < other.time();
}

/// Dump object in human-readable format
void
Hdf5EventId::print(std::ostream& os) const
{
  os << "Hdf5EventId(run=" << m_run << ", time=" << m_time << ')';
}

} // namespace PSHdf5Input
