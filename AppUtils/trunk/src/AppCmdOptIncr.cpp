//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdOptIncr
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
#include "AppUtils/AppCmdOptIncr.h"

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

AppCmdOptIncr::AppCmdOptIncr(const std::string& optNames, const std::string& descr, int defValue)
  : AppCmdOptBase(optNames, "(incr)", descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptIncr::AppCmdOptIncr(AppCmdLine& parser, const std::string& optNames, const std::string& descr, int defValue)
  : AppCmdOptBase(optNames, "(incr)", descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
  parser.addOption(*this);
}

// Destructor
AppCmdOptIncr::~AppCmdOptIncr( )
{
}

/**
 *  Returns true if option requires argument. Does not make sense for
 *  positional arguments.
 */
bool
AppCmdOptIncr::hasArgument() const
{
  return false ;
}

/**
 *  Set the value of the argument.
 */
void
AppCmdOptIncr::setValue ( const std::string& value )
{
  ++ _value ;
  _changed = true ;
}

/**
 *  True if the value of the option was changed from command line.
 */
bool
AppCmdOptIncr::valueChanged () const
{
  return _changed ;
}

/**
 *  Return current value of the argument
 */
int
AppCmdOptIncr::value() const
{
  return _value ;
}

/**
 *  reset option to its default value
 */
void
AppCmdOptIncr::reset()
{
  _value = _defValue ;
  _changed = false ;
}

} // namespace AppUtils
