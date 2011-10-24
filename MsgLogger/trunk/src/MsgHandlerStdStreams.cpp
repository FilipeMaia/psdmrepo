//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MsgHandlerStdStreams
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
#include "MsgLogger/MsgHandlerStdStreams.h"

//-------------
// C Headers --
//-------------
extern "C" {
}

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgFormatter.h"
#include "MsgLogger/MsgLogLevel.h"
#include "MsgLogger/MsgLogRecord.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace MsgLogger {

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

// Constructor
MsgHandlerStdStreams::MsgHandlerStdStreams()
  : MsgHandler()
  , m_mutex()
{
}

// Destructor
MsgHandlerStdStreams::~MsgHandlerStdStreams()
{
}

/// get the stream for the specified log level
bool
MsgHandlerStdStreams::log ( const MsgLogRecord& record ) const
{
  if ( ! logging( record.level() ) ) {
    return false ;
  }

  static bool threadSafe = getenv("MSGLOGTHREADSAFE") ;
  if (threadSafe) m_mutex.lock() ;

  if ( record.level() <= MsgLogLevel::info ) {
    formatter().format ( record, std::cout ) ;
    std::cout << std::endl ;
  } else {
    formatter().format ( record, std::cerr ) ;
    std::cerr << std::endl ;
  }

  if (threadSafe) m_mutex.unlock() ;

  return true ;
}

} // namespace MsgLogger
