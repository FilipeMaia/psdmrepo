//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcException...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "odbcpp/OdbcException.h"

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

namespace odbcpp {


std::string
OdbcException::h2msg ( SQLHANDLE* h, int typecode, const char* context )
{
  std::ostringstream str ;
  SQLINTEGER i = 0;
  SQLINTEGER native;
  SQLCHAR state[7];
  SQLCHAR text[256];
  SQLSMALLINT len;
  SQLRETURN ret;
  do {
    text[0] = '\0' ;
    ret = SQLGetDiagRec( typecode, *h, ++i, state, &native, text, sizeof(text), &len );
    if (SQL_SUCCEEDED(ret)) {
      if ( i > 1 ) str << '\n' ;
      str << context << " [" << state << "/" << native << "]" << text ;
    }
  } while( false /*ret != SQL_NO_DATA*/ );

  return str.str() ;
}


} // namespace odbcpp
