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

namespace {

std::string optspec(const std::string& optname)
{
  if (optname.size() == 1) return "-" + optname;
  return "--" + optname;
}

}

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

AppCmdLexicalCastFromStringException::AppCmdLexicalCastFromStringException ( const std::string& string, const std::string& message )
  : AppCmdException( "failed to convert string \""+string+"\" to value: "+message )
{
}

AppCmdLexicalCastToStringException::AppCmdLexicalCastToStringException ( const std::string& message )
  : AppCmdException( "failed to convert value to string: "+message )
{
}

AppCmdOptDefinedException::AppCmdOptDefinedException ( const std::string& option )
  : AppCmdException( "option '" + ::optspec(option) + "' is already defined" )
{
}

AppCmdOptUnknownException::AppCmdOptUnknownException ( const std::string& option )
  : AppCmdException( "option '" + ::optspec(option) + "' is unknown" )
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

AppCmdArgListTooLong::AppCmdArgListTooLong()
  : AppCmdException( "command line argument list is too long" )
{
}

AppCmdArgListTooShort::AppCmdArgListTooShort()
  : AppCmdException( "missing positional required argument(s)" )
{
}

} // namespace AppUtils
