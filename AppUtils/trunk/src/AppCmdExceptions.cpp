//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdExceptions...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppCmdExceptions.h"

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

namespace AppUtils {

AppCmdException::AppCmdException ( const std::string& msg )
  : std::runtime_error( msg )
{
}

AppCmdTypeCvtException::AppCmdTypeCvtException ( const std::string& string, const std::string& type )
  : AppCmdException( "failed to convert string \""+string+"\" to type "+type )
{
}

AppCmdOptReservedException::AppCmdOptReservedException ( char option )
  : AppCmdException( std::string("short option '-")+option+"' is reserved" )
{
}
AppCmdOptReservedException::AppCmdOptReservedException ( const std::string& option )
  : AppCmdException( "long option '--"+option+"' is reserved" )
{
}

AppCmdOptDefinedException::AppCmdOptDefinedException ( char option )
  : AppCmdException( std::string("short option '-")+option+"' is already defined" )
{
}
AppCmdOptDefinedException::AppCmdOptDefinedException ( const std::string& option )
  : AppCmdException( "long option '--"+option+"' is already defined" )
{
}

AppCmdOptUnknownException::AppCmdOptUnknownException ( char option )
  : AppCmdException( std::string("option '-")+option+"' is unknown" )
{
}
AppCmdOptUnknownException::AppCmdOptUnknownException ( const std::string& option )
  : AppCmdException( "long option '--"+option+"' is unknown" )
{
}

AppCmdArgOrderException::AppCmdArgOrderException ( const std::string& arg )
  : AppCmdException( "cannot add required argument after non-required: "+arg )
{
}

AppCmdOptNameException::AppCmdOptNameException ( const std::string& option )
  : AppCmdException( "option name cannot start with dash: "+option )
{
}

} // namespace AppUtils
