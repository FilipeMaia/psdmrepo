//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Hdf5Utils...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSHdf5Input/Hdf5Utils.h"

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

// get value of time attribute.
PSTime::Time
Hdf5Utils::getTime(hdf5pp::Group& grp, const std::string& time)
{
  uint32_t sec = getAttr(grp, time+".seconds", uint32_t(0));
  uint32_t nsec = getAttr(grp, time+".nanoseconds", uint32_t(0));
  return PSTime::Time(sec, nsec);
}

} // namespace PSHdf5Input
