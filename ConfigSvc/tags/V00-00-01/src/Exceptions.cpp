//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Exceptions...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigSvc/Exceptions.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/lexical_cast.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace ConfigSvc {

//----------------
// Constructors --
//----------------
Exception::Exception( const std::string& what )
  : std::runtime_error( "ConfigSvc::Exception: " + what )
{
}

ExceptionInitialized::ExceptionInitialized () 
  : Exception("configuration service has been initialized already")
{
}

ExceptionSyntax::ExceptionSyntax ( const std::string& file, 
    int lineno, 
    const std::string& what ) 
  : Exception( "config file syntax error: " + file + ":" + 
               boost::lexical_cast<std::string>(lineno) + ": " + what)
{
  
}

ExceptionNotInitialized::ExceptionNotInitialized () 
  : Exception("configuration service was not properly initialized")
{
}

ExceptionFileMissing::ExceptionFileMissing ( const std::string& file ) 
  : Exception("configuration file missing or unreadable: " + file)
{
}

ExceptionFileRead::ExceptionFileRead ( const std::string& file )
  : Exception("configuration file read error: " + file)
{
}

ExceptionMissing::ExceptionMissing ( const std::string& section, 
                                     const std::string& parameter )
  : Exception( "parameter is not found: " + parameter + " (in section " + section + ")")
{
}

ExceptionCvtFail::ExceptionCvtFail ( const std::string& string )
  : Exception("conversion from string failed for value: " + string)
{
}

} // namespace ConfigSvc
