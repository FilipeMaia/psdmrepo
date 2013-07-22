//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdOptToggle
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

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppCmdOptToggle.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdLine.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace AppUtils {

AppCmdOptToggle::AppCmdOptToggle(const std::string& optNames, const std::string& descr, bool defValue)
  : AppCmdOptBase(optNames, "(toggle)", descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptToggle::AppCmdOptToggle(AppCmdLine& parser, const std::string& optNames, const std::string& descr,
    bool defValue)
  : AppCmdOptBase(optNames, "(toggle)", descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
  parser.addOption(*this);
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
 *  Set the value of the argument.
 *
 *  @return The number of consumed words. If it is negative then error has occured.
 */
void
AppCmdOptToggle::setValue ( const std::string& value )
{
  _value = ! _value ;
  _changed = true ;
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
