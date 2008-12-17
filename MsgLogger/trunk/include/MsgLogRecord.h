#ifndef MSGLOGRECORD_HH
#define MSGLOGRECORD_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MsgLogRecord.
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
#include "MsgLogger/MsgLogger.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Clkass which defines a single logging message (record.) It has such attributes
 *  as message itself, logging level, corresponding logger name, file/line where
 *  the message originated. For performance optimization purposes the message is
 *  passed as a pointer to the streambuf. This complicates things a bit, you have
 *  to be careful when extracting message test from the streambuf, but avoids
 *  copying of the strings. Also for optimization reasons the timestamp is not a part
 *  of the message, but is added to the formatted message only during formatting
 *  (only if needed.)
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see MsgLogRecordMsgLogRecord
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

namespace MsgLogger {

class MsgLoggerImpl ;

class MsgLogRecord {

public:

  // Construct root logger
  MsgLogRecord( const std::string& logger,
                MsgLogLevel level,
                const char* fileName,
                int linenum,
                std::streambuf* msgbuf )
    : _logger(logger), _level(level), _fileName(fileName), _lineNum(linenum), _msgbuf(msgbuf)
    {}

  // Destructor
  ~MsgLogRecord() {}

  /// get logger name
  const std::string& logger() const { return _logger ; }

  /// get message log level
  MsgLogLevel level() const { return _level ; }

  /// get message location
  const char* fileName() const { return _fileName ; }
  int lineNum() const { return _lineNum ; }

  /// get the stream for the specified log level
  std::streambuf* msgbuf() const { return _msgbuf ; }

protected:

  // Helper functions

private:

  // Friends

  // Data members
  const std::string& _logger ;
  const MsgLogLevel _level ;
  const char* _fileName ;
  int _lineNum ;
  std::streambuf* _msgbuf ;

  MsgLogRecord( const MsgLogRecord& );
  MsgLogRecord& operator= ( const MsgLogRecord& );

};
} // namespace MsgLogger

#endif // MSGLOGRECORD_HH
