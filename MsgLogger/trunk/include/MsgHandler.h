#ifndef MSGHANDLER_HH
#define MSGHANDLER_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MsgHandler.
//
// Environment:
//      This software was developed for the BaBar collaboration.  If you
//      use all or part of it, please give an appropriate acknowledgement.
//
// Author List:
//      Andy Salnikov
//
// Copyright Information:
//      Copyright (C) 2005 SLAC
//
//------------------------------------------------------------------------

//-------------
// C Headers --
//-------------
extern "C" {
}

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogLevel.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  The base class for handling messages. Handling means dispatching
 *  messages to their final distantion, such as terminal, log file,
 *  external logging service, etc. Handlers like loggers have corresponding
 *  logging level which can be changed by clients. Default logging
 *  level for handlers is debug, which means they will log whatever
 *  loggers give them. For formatting the messages handlers use
 *  MsgFormatter class, clients can attach their own formatter objects to
 *  handlers, otherwise default formatter class will be used.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see MsgLogger
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

namespace MsgLogger {

class MsgFormatter ;
class MsgLogRecord ;

class MsgHandler {

public:

  // Destructor
  virtual ~MsgHandler() ;

  /// attaches the formatter, will be owned by handler
  virtual void setFormatter ( MsgFormatter* formatter ) ;

  /// set the logger level, messages with the level below this won't be logged
  virtual void setLevel ( MsgLogLevel level ) ;

  /// check if the specified level will log any message
  virtual bool logging ( MsgLogLevel sev ) const ;

  /// get the stream for the specified log level
  virtual bool log ( const MsgLogRecord& record ) const = 0 ;

protected:

  // Constructor
  MsgHandler() ;

  // get the formatter
  MsgFormatter& formatter() const ;

private:

  // Data members
  mutable MsgFormatter* _formatter ;
  MsgLogLevel _level ;

  // Disable copy
  MsgHandler( const MsgHandler& );
  MsgHandler& operator= ( const MsgHandler& );

};
} // namespace MsgLogger

#endif // MSGHANDLER_HH
