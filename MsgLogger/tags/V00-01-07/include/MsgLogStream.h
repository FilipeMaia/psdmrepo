#ifndef MSGLOGSTREAM_HH
#define MSGLOGSTREAM_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MsgLogStream.
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
#include <string>
#include <sstream>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogLevel.h"
#include "MsgLogger/MsgLogger.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Macros(grrr!) for user's convenience
 */
#ifdef MsgLog
#undef MsgLog
#endif
/**
 *  @def MsgLog(logger,sev,msg)
 *
 *  @brief Macro which sends single message to a named logger in logging service.
 *
 *  @param logger   Name of the logger
 *  @param sev      Severity level (one of debug, trace, info, warning, error)
 *  @param msg      Message, anything that can appear on the right side of << operator.
 *
 *  This macro provides convenience method for working with the messaging facility.
 *  If the logging level configured by application for the given logger name (first argument)
 *  allows messages of the given severity (second argument) then everything included in
 *  the last argument is formatted via << insertion operator and sent to logging service.
 *  Brief example:
 *  @code
 *  MsgLog("logger", debug, "key = " << key << " value = " << value << " count = " << count);
 *  @endcode
 */
#define MsgLog(logger,sev,msg) \
  if ( MsgLogger::MsgLogger(logger).logging(MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev)) ) { \
    MsgLogger::MsgLogStream _msg_log_stream_123(logger, MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev), __FILE__, __LINE__) ;\
    _msg_log_stream_123.ostream_hack() << msg ; \
  }

#ifdef MsgLogRoot
#undef MsgLogRoot
#endif
/**
 *  @def MsgLogRoot(sev,msg)
 *
 *  @brief Macro which sends single message to a root logger in logging service.
 *
 *  @param sev      Severity level (one of debug, trace, info, warning, error)
 *  @param msg      Message, anything that can appear on the right side of << operator.
 *
 *  See @c MsgLog for details. Brief example:
 *  @code
 *  MsgLogRoot(debug, "key = " << key << " value = " << value << " count = " << count);
 *  @endcode
 */
#define MsgLogRoot(sev,msg) \
  if ( MsgLogger::MsgLogger().logging(MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev)) ) { \
    MsgLogger::MsgLogStream _msg_log_stream_123(MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev), __FILE__, __LINE__) ;\
    _msg_log_stream_123.ostream_hack() << msg ; \
  }

#ifdef WithMsgLog
#undef WithMsgLog
#endif
/**
 *  @def WithMsgLog(logger,sev,str)
 *
 *  @brief Macro which provides scoped logging stream with output sent to a named logger.
 *
 *  @param logger   Name of the logger
 *  @param sev      severity level (one of debug, trace, info, warning, error)
 *  @param str      stream
 *
 *  The use of this macro must be followed by the compound statement (enclosed in {}). The stream defined by this
 *  macro (last argument) is available inside the compound statement and can be used multiple times. All output to
 *  this stream is collected and is sent to the messaging service at the end of the scope of the compound statement;
 *  complete output appears as a single message.  This macro is useful when output is produced inside loop or
 *  if/then/else statement. Brief example:
 *  @code
 *  WithMsgLog("logger", debug, out) {
 *    if (condition) out << "condition is true, ";
 *    out << "values: ";
 *    for (int i = 0; i < max; ++ i) out << ' ' << value[i];
 *  }
 *  @endcode
 */
#define WithMsgLog(logger,sev,str) \
  for ( MsgLogger::MsgLogStream str(logger, MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev), __FILE__ , __LINE__) ; str.ok() ; str.finish() )

#ifdef WithMsgLogRoot
#undef WithMsgLogRoot
#endif
/**
 *  @def WithMsgLogRoot(sev,str)
 *
 *  @brief Macro which provides scoped logging stream with output sent to a root logger.
 *
 *  @param sev      severity level (one of debug, trace, info, warning, error)
 *  @param str      stream
 *
 *  See @c WithMsgLog for details. Brief example:
 *  @code
 *  WithMsgLogRoot(debug, out) {
 *    if (condition) out << "condition is true, ";
 *    out << "values: ";
 *    for (int i = 0; i < max; ++ i) out << ' ' << value[i];
 *  }
 *  @endcode
 */
#define WithMsgLogRoot(sev,str) \
  for ( MsgLogger::MsgLogStream str(MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev), __FILE__ , __LINE__) ; str.ok() ; str.finish() )

namespace MsgLogger {

/**
 *  @ingroup MsgLogger
 *
 *  Special stream class (subclass of standard stream class) which collects
 *  the message source and forwards complete message to the logger class
 *  on destruction.
 *
 *  This software was developed originally for the BaBar collaboration and
 *  adapted/rewritten for LUSI.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see MsgLogger
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class MsgLogStream : public std::stringstream {

public:

  /**
   *  Constructors. 'file' argument is usually a filenamestring constructed from
   *  __FILE__ macros. It is char* type instead of std::string for optimization
   *  reasons (crappy C++ has no compile-time constructors for classes.) The pointer is stored
   *  in an object, content is not copied, so make sure that you don't overwrite the string you pass.
   */
  explicit MsgLogStream ( MsgLogLevel sev, const char* file = 0, int line = -1 ) ;
  MsgLogStream ( const std::string& loggerName, MsgLogLevel sev, const char* file = 0, int line = -1 ) ;

  // Destructor
  virtual ~MsgLogStream() ;

  // g++ somehow fails to recognize temporary MsgLogStream() as a good stream,
  // had to add this "cast" operation
  std::ostream& ostream_hack() { return *this ; }

  // send my content to logger
  void emit() ;

  // get the state of the stream
  bool ok() const { return _ok ; }

  // set the state of the stream to "not OK"
  void finish() { _ok = false ; }

protected:

private:

  // Data members
  std::string _logger ;
  MsgLogLevel _sev ;
  const char* _file ;
  int _lineNum ;
  bool _ok ;

  // Note: if your class needs a copy constructor or an assignment operator,
  //  make one of the following public and implement it.
  MsgLogStream( const MsgLogStream& );                // Copy Constructor
  MsgLogStream& operator= ( const MsgLogStream& );    // Assignment op

};

} // namespace MsgLogger

#endif // MSGLOGSTREAM_HH
