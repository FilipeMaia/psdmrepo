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
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "AppUtils/AppCmdOptIncr.h"

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
AppCmdOptIncr::AppCmdOptIncr ( char shortOpt,
			       const std::string& longOpt,
			       const std::string& descr,
			       int defValue )
  : AppCmdOptBase()
  , _shortOpt(shortOpt)
  , _longOpt(longOpt)
  , _name("(incr)")
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptIncr::AppCmdOptIncr ( const std::string& longOpt,
                               const std::string& descr,
                               int defValue )
  : AppCmdOptBase()
  , _shortOpt('\0')
  , _longOpt(longOpt)
  , _name("(incr)")
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

AppCmdOptIncr::AppCmdOptIncr ( char shortOpt,
                               const std::string& descr,
                               int defValue )
  : AppCmdOptBase()
  , _shortOpt(shortOpt)
  , _longOpt()
  , _name("(incr)")
  , _descr(descr)
  , _value(defValue)
  , _defValue(defValue)
  , _changed(false)
{
}

// Destructor
AppCmdOptIncr::~AppCmdOptIncr( ) throw()
{
}

/**
 *  Returns true if option requires argument. Does not make sense for
 *  positional arguments.
 */
bool
AppCmdOptIncr::hasArgument() const throw()
{
  return false ;
}

/**
 *  Get the name of the parameter
 */
const std::string&
AppCmdOptIncr::name() const throw()
{
  return _name ;
}

/**
 *  Get one-line description
 */
const std::string&
AppCmdOptIncr::description() const throw()
{
  return _descr ;
}

/**
 *  Return short option symbol for -x option
 */
char
AppCmdOptIncr::shortOption() const throw()
{
  return _shortOpt ;
}

/**
 *  Return long option symbol for --xxxxx option
 */
const std::string&
AppCmdOptIncr::longOption() const throw()
{
  return _longOpt ;
}

/**
 *  Set the value of the argument.
 */
void
AppCmdOptIncr::setValue ( const std::string& value ) throw(AppCmdException)
{
  ++ _value ;
  _changed = true ;
}

/**
 *  True if the value of the option was changed from command line.
 */
bool
AppCmdOptIncr::valueChanged () const throw()
{
  return _changed ;
}

/**
 *  Return current value of the argument
 */
int
AppCmdOptIncr::value() const throw()
{
  return _value ;
}

/**
 *  reset option to its default value
 */
void
AppCmdOptIncr::reset() throw()
{
  _value = _defValue ;
  _changed = false ;
}

} // namespace AppUtils
