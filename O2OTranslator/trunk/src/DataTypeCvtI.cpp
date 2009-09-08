//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2ODataTypeCvtI...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/DataTypeCvtI.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

//--------------
// Destructor --
//--------------
DataTypeCvtI::~DataTypeCvtI ()
{
}

// generate the group name for the child folder
std::string
DataTypeCvtI::cvtGroupName( const std::string& grpName, const Pds::DetInfo& info )
{
  std::ostringstream str ;
  str << grpName << '/' << Pds::DetInfo::name(info.detector()) << '.' << info.detId()
      << ':' << Pds::DetInfo::name(info.device()) << '.' << info.devId() ;
  return str.str() ;
}
} // namespace O2OTranslator
