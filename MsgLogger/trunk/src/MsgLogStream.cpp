//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MsgLogStream
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
#include "MsgLogger/MsgLogStream.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <cstdlib>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "MsgLogger/MsgLogRecord.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace MsgLogger {

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

MsgLogStream::MsgLogStream ( MsgLogLevel sev, const char* file, int line )
  : std::stringstream()
  , _logger()
  , _sev(sev)
  , _file(file)
  , _lineNum(line)
{
  _ok = MsgLogger().logging(_sev) ;
}
MsgLogStream::MsgLogStream ( const std::string& loggerName, MsgLogLevel sev, const char* file, int line )
  : std::stringstream()
  , _logger(loggerName)
  , _sev(sev)
  , _file(file)
  , _lineNum(line)
{
  _ok = MsgLogger(_logger).logging(_sev) ;
}

// Destructor
MsgLogStream::~MsgLogStream()
{
  emit() ;
}

// send my content to logger
void
MsgLogStream::emit()
{
  // check if we need to send it at all
  MsgLogger logger( _logger ) ;
  if ( logger.logging ( _sev ) ) {
    MsgLogRecord record ( _logger, _sev, _file, _lineNum, rdbuf() ) ;
    logger.log ( record ) ;
  }
  if ( _sev == MsgLogLevel::fatal ) {
    abort() ;
  }
}

} // namespace MsgLogger

