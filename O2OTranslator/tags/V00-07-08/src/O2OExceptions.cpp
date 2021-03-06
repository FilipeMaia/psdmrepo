//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OExceptions...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/O2OExceptions.h"

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

namespace O2OTranslator {

//----------------
// Constructors --
//----------------
O2OException::O2OException (const std::string& className, const std::string& what)
  : std::runtime_error( className+": "+what)
{
}

} // namespace O2OTranslator
