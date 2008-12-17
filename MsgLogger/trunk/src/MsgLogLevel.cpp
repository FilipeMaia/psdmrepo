//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MsgLogLevel
//
// Environment:
//	Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Andy Salnikov
//
// Copyright Information:
//      Copyright (C) 2005 SLAC
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "MsgLogger/MsgLogLevel.h"

//---------------
// C++ Headers --
//---------------
#include <stdexcept>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace MsgLogger {

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

MsgLogLevel::MsgLogLevel( int code )
{
  if ( code < int(debug) ) {
    code = int(debug) ;
  } else if ( code > int(error) ) {
    code = int(error) ;
  }
  _level = Level(code) ;
}

MsgLogLevel::MsgLogLevel( const std::string& level )
{
  if ( level == "trace" ) {
    _level = trace ;
  } else if ( level == "debug" ) {
    _level = debug ;
  } else if ( level == "warning" ) {
    _level = warning ;
  } else if ( level == "info" ) {
    _level = info ;
  } else if ( level == "error" ) {
    _level = error ;
  } else if ( level == "fatal" ) {
    _level = fatal ;
  } else if ( level == "nolog" ) {
    _level = nolog ;
  } else {
    throw std::out_of_range ( "unexpected logging level name" ) ;
  }
}

// get full printable name of Level level
const char*
MsgLogLevel::levelName () const
{
  switch ( _level ) {
    case debug:
      return "debug" ;
    case trace:
      return "trace" ;
    case info:
      return "info" ;
    case warning:
      return "warning" ;
    case error:
      return "error" ;
    case fatal:
      return "fatal" ;
    case nolog:
    default:
      return "no-log" ;
  }
}

// get 3-letter printable name of Level code
const char*
MsgLogLevel::level3 () const
{
  switch ( _level ) {
    case debug:
      return "DBG" ;
    case trace:
      return "TRC" ;
    case info:
      return "INF" ;
    case warning:
      return "WRN" ;
    case error:
      return "ERR" ;
    case fatal:
      return "FTL" ;
    case nolog:
    default:
      return "???" ;
  }
}

// get one-char Level code
char
MsgLogLevel::levelLetter () const
{
  switch ( _level ) {
    case debug:
      return 'D' ;
    case trace:
      return 'T' ;
    case info:
      return 'I' ;
    case warning:
      return 'W' ;
    case error:
      return 'E' ;
    case fatal:
      return 'F' ;
    case nolog:
    default:
      return '?' ;
  }
}

} // namespace MsgLogger
