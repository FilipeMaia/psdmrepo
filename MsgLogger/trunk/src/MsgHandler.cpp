//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MsgHandler
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

//-----------------------
// This Class's Header --
//-----------------------
#include "MsgLogger/MsgHandler.h"

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
#include "MsgLogger/MsgFormatter.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace MsgLogger {

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

// Constructor
MsgHandler::MsgHandler()
  : _formatter(0)
  , _level(MsgLogLevel::debug)
{
}

// Destructor
MsgHandler::~MsgHandler()
{
  delete _formatter ;
}

/// attaches the formatter, will be owned by handler
void
MsgHandler::setFormatter ( MsgFormatter* formatter )
{
  delete _formatter ;
  _formatter = formatter ;
}

/// set the logger level, messages with the level below this won't be logged
void
MsgHandler::setLevel ( MsgLogLevel level )
{
  _level = level ;
}

/// check if the specified level will log any message
bool
MsgHandler::logging ( MsgLogLevel level ) const
{
  return level >= _level ;
}

// get the formatter
MsgFormatter&
MsgHandler::formatter() const
{
  if ( ! _formatter ) {
    _formatter = new MsgFormatter ;
  }
  return *_formatter ;
}

} // namespace MsgLogger

