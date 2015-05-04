//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MergeMode...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/MergeMode.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

// Make merge mode from string
MergeMode 
mergeMode(const std::string& str)
{
  if (str == "MergeFileName" or str == "FileName") {
    return MergeFileName;
  } else if (str == "MergeOneStream" or str == "OneStream") {
    return MergeOneStream;
  } else if (str == "MergeNoChunking" or str == "NoChunking") {
    return MergeNoChunking;
  } else {
    throw InvalidMergeMode(ERR_LOC, str);
  }

}

std::ostream&
operator<<(std::ostream& out, MergeMode mode)
{
  const char* str = "*ERROR*";
  switch(mode) {
  case MergeOneStream:
    str = "MergeOneStream";
    break;
  case MergeNoChunking:
    str = "MergeNoChunking";
    break;
  case MergeFileName:
    str = "MergeFileName";
    break;
  }
  return out << str;
}

} // namespace XtcInput
