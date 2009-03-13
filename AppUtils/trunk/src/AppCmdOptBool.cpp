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
#include "Lusi/Lusi.h"

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
  : AppCmdOptBase()
  , _shortOpt(shortOpt)
  , _longOpt(longOpt)
  , _name()
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptBool::AppCmdOptBool ( const std::string& longOpt,
                               const std::string& descr,
                               bool defValue )
  : AppCmdOptBase()
  , _shortOpt('\0')
  , _longOpt(longOpt)
  , _name()
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptBool::AppCmdOptBool ( char shortOpt,
                               const std::string& descr,
                               bool defValue )
  : AppCmdOptBase()
  , _shortOpt(shortOpt)
  , _longOpt()
  , _name()
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

// Destructor
AppCmdOptBool::~AppCmdOptBool( ) throw()
{
}

/**
 *  Returns true if option requires argument. Does not make sense for
 *  positional arguments.
 */
bool
AppCmdOptBool::hasArgument() const throw()
{
  return false ;
}

/**
 *  Get the name of the paramater
 */
const std::string&
AppCmdOptBool::name() const throw()
{
  return _name ;
}

/**
 *  Get one-line description
 */
const std::string&
AppCmdOptBool::description() const throw()
{
  return _descr ;
}

/**
 *  Return short option symbol for -x option
 */
char
AppCmdOptBool::shortOption() const throw()
{
  return _shortOpt ;
}

/**
 *  Return long option symbol for --xxxxx option
 */
const std::string&
AppCmdOptBool::longOption() const throw()
{
  return _longOpt ;
}

/**
 *  Set the value of the argument.
 *
 *  @return The number of consumed words. If it is negative then error has occured.
 */
void
AppCmdOptBool::setValue ( const std::string& value ) throw(AppCmdException)
{
  _value = ! _defValue ;
  _changed = true ;
}

/**
 *  True if the value of the option was changed from command line.
 */
bool
AppCmdOptBool::valueChanged () const throw()
{
  return _changed ;
}

/**
 *  Return current value of the argument
 */
bool
AppCmdOptBool::value() const throw()
{
  return _value ;
}

/**
 *  reset option to its default value
 */
void
AppCmdOptBool::reset() throw()
{
  _value = _defValue ;
  _changed = false ;
}

} // namespace AppUtils
