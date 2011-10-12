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
#include "Lusi/Lusi.h"

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
AppCmdOptSize::AppCmdOptSize ( char shortOpt,
                               const std::string& longOpt,
                               const std::string& name,
                               const std::string& descr,
                               value_type defValue )
  : AppCmdOptBase()
  , _shortOpt(shortOpt)
  , _longOpt(longOpt)
  , _name(name)
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptSize::AppCmdOptSize ( const std::string& longOpt,
                               const std::string& name,
                               const std::string& descr,
                               value_type defValue )
  : AppCmdOptBase()
  , _shortOpt('\0')
  , _longOpt(longOpt)
  , _name(name)
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptSize::AppCmdOptSize ( char shortOpt,
                               const std::string& name,
                               const std::string& descr,
                               value_type defValue )
  : AppCmdOptBase()
  , _shortOpt(shortOpt)
  , _longOpt()
  , _name(name)
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

// Destructor
AppCmdOptSize::~AppCmdOptSize( ) throw()
{
}

/**
 *  Returns true if option requires argument. Does not make sense for
 *  positional arguments.
 */
bool
AppCmdOptSize::hasArgument() const throw()
{
  return true ;
}

/**
 *  Get the name of the parameter
 */
const std::string&
AppCmdOptSize::name() const throw()
{
  return _name ;
}

/**
 *  Get one-line description
 */
const std::string&
AppCmdOptSize::description() const throw()
{
  return _descr ;
}

/**
 *  Return short option symbol for -x option
 */
char
AppCmdOptSize::shortOption() const throw()
{
  return _shortOpt ;
}

/**
 *  Return long option symbol for --xxxxx option
 */
const std::string&
AppCmdOptSize::longOption() const throw()
{
  return _longOpt ;
}

/**
 *  Set the value of the argument.
 */
void
AppCmdOptSize::setValue ( const std::string& value ) throw(AppCmdException)
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
}

/**
 *  True if the value of the option was changed from command line.
 */
bool
AppCmdOptSize::valueChanged () const throw()
{
  return _changed ;
}

/**
 *  Return current value of the argument
 */
AppCmdOptSize::value_type
AppCmdOptSize::value() const throw()
{
  return _value ;
}

/**
 *  reset option to its default value
 */
void
AppCmdOptSize::reset() throw()
{
  _value = _defValue ;
  _changed = false ;
}

} // namespace AppUtils
