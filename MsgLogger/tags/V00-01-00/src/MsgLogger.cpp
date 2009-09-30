//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MsgLogger
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
#include "MsgLogger/MsgLogger.h"

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
#include "MsgLogger/MsgLoggerImpl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace MsgLogger {

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

// Construct a logger
MsgLogger::MsgLogger( const std::string& name )
  : _impl ( MsgLoggerImpl::getLogger(name) )
{
}

/// set the logger level, messages with the level below this won't be logged
void
MsgLogger::setLevel ( MsgLogLevel level )
{
  if ( _impl ) _impl->setLevel ( level ) ;
}

/// define wheter or not we need messages propagated to ancestors
void
MsgLogger::propagate ( bool flag )
{
  if ( _impl ) _impl->propagate( flag ) ;
}

/// add a handler for the messages, takes ownership of the object
void
MsgLogger::addHandler ( MsgHandler* handler )
{
  if ( _impl ) _impl->addHandler ( handler ) ;
}


/// check if the specified level will log any message
bool
MsgLogger::logging ( MsgLogLevel sev ) const
{
  return _impl ? _impl->logging ( sev ) : false ;
}

/// get the stream for the specified log level
bool
MsgLogger::log ( const MsgLogRecord& record ) const
{
  return _impl ? _impl->log ( record ) : false ;
}


} // namespace MsgLogger
