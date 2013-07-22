//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdOptSize...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppCmdOptSize.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <cstdlib>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdExceptions.h"
#include "AppUtils/AppCmdOptGroup.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace AppUtils {

/**
 *  Ctor
 */
AppCmdOptSize::AppCmdOptSize(const std::string& optNames, const std::string& name, const std::string& descr,
    value_type defValue)
  : AppCmdOptBase(optNames, name, descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptSize::AppCmdOptSize(AppCmdOptGroup& group, const std::string& optNames, const std::string& name,
    const std::string& descr, value_type defValue)
  : AppCmdOptBase(optNames, name, descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
  group.addOption(*this);
}

// Destructor
AppCmdOptSize::~AppCmdOptSize( )
{
}

/**
 *  Returns true if option requires argument. Does not make sense for
 *  positional arguments.
 */
bool
AppCmdOptSize::hasArgument() const
{
  return true ;
}

/**
 *  Set the value of the argument.
 */
void
AppCmdOptSize::setValue ( const std::string& value )
{
  char* eptr ;
  value_type tmp = std::strtoull ( value.c_str(), &eptr, 0 ) ;
  switch ( *eptr ) {
  case 'G' :
    tmp *= 1024 ;
  case 'M' :
    tmp *= 1024 ;
  case 'k' :
  case 'K' :
    tmp *= 1024 ;
    ++ eptr ;
  case '\0' :
    break ;
  }

  if ( *eptr != '\0' ) throw AppCmdTypeCvtException ( value, "size" ) ;

  _value = tmp ;
  _changed = true ;
}

/**
 *  True if the value of the option was changed from command line.
 */
bool
AppCmdOptSize::valueChanged () const
{
  return _changed ;
}

/**
 *  Return current value of the argument
 */
AppCmdOptSize::value_type
AppCmdOptSize::value() const
{
  return _value ;
}

/**
 *  reset option to its default value
 */
void
AppCmdOptSize::reset()
{
  _value = _defValue ;
  _changed = false ;
}

} // namespace AppUtils
