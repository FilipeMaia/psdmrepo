//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdOptToggleToggle
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//	Andy Salnikov		originator
//
// Copyright Information:
//	Copyright (C) 2003	SLAC
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppCmdOptToggle.h"

//-------------
// C Headers --
//-------------
extern "C" {
}

//---------------
// C++ Headers --
//---------------

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

/**
 *  Ctor
 */
AppCmdOptToggle::AppCmdOptToggle ( char shortOpt,
				   const std::string& longOpt,
				   const std::string& descr,
				   bool defValue )
  : AppCmdOptBase()
  , _shortOpt(shortOpt)
  , _longOpt(longOpt)
  , _name("(toggle)")
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

// Destructor
AppCmdOptToggle::~AppCmdOptToggle( )
{
}

/**
 *  Returns true if option requires argument. Does not make sense for
 *  positional arguments.
 */
bool
AppCmdOptToggle::hasArgument() const
{
  return false ;
}

/**
 *  Get the name of the paramater
 */
const std::string&
AppCmdOptToggle::name() const
{
  return _name ;
}

/**
 *  Get one-line description
 */
const std::string&
AppCmdOptToggle::description() const
{
  return _descr ;
}

/**
 *  Return short option symbol for -x option
 */
char
AppCmdOptToggle::shortOption() const
{
  return _shortOpt ;
}

/**
 *  Return long option symbol for --xxxxx option
 */
const std::string&
AppCmdOptToggle::longOption() const
{
  return _longOpt ;
}

/**
 *  Set the value of the argument.
 *
 *  @return The number of consumed words. If it is negative then error has occured.
 */
bool
AppCmdOptToggle::setValue ( const std::string& value )
{
  _value = ! _value ;
  _changed = true ;
  return true ;
}

/**
 *  True if the value of the option was changed from command line.
 */
bool
AppCmdOptToggle::valueChanged () const
{
  return _changed ;
}

/**
 *  Return current value of the argument
 */
bool
AppCmdOptToggle::value() const
{
  return _value ;
}

/**
 *  reset option to its default value
 */
void
AppCmdOptToggle::reset()
{
  _value = _defValue ;
  _changed = false ;
}

} // namespace AppUtils
