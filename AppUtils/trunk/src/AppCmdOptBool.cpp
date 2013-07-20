//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AppCmdOptBool
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
#include "AppUtils/AppCmdOptBool.h"

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
AppCmdOptBool::AppCmdOptBool ( char shortOpt,
                               const std::string& longOpt,
                               const std::string& descr,
                               bool defValue )
  : AppCmdOptBase(longOpt+","+std::string(1, shortOpt), "", descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptBool::AppCmdOptBool ( const std::string& optNames,
                               const std::string& descr,
                               bool defValue )
  : AppCmdOptBase(optNames, "", descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptBool::AppCmdOptBool ( char shortOpt,
                               const std::string& descr,
                               bool defValue )
  : AppCmdOptBase(std::string(1, shortOpt), "", descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

// Destructor
AppCmdOptBool::~AppCmdOptBool( )
{
}

/**
 *  Returns true if option requires argument. Does not make sense for
 *  positional arguments.
 */
bool
AppCmdOptBool::hasArgument() const
{
  return false ;
}

/**
 *  Set the value of the argument.
 *
 *  @return The number of consumed words. If it is negative then error has occured.
 */
void
AppCmdOptBool::setValue ( const std::string& value )
{
  _value = ! _defValue ;
  _changed = true ;
}

/**
 *  True if the value of the option was changed from command line.
 */
bool
AppCmdOptBool::valueChanged () const
{
  return _changed ;
}

/**
 *  Return current value of the argument
 */
bool
AppCmdOptBool::value() const
{
  return _value ;
}

/**
 *  reset option to its default value
 */
void
AppCmdOptBool::reset()
{
  _value = _defValue ;
  _changed = false ;
}

} // namespace AppUtils
