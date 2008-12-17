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
#define MsgLog(logger,sev,msg) \
  if ( MsgLogger::MsgLogger(logger).logging(MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev)) ) { \
    MsgLogger::MsgLogStream stream(logger, MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev), __FILE__, __LINE__) ;\
    stream.ostream_hack() << msg ; \
  }

#ifdef MsgLogRoot
#undef MsgLogRoot
#endif
#define MsgLogRoot(sev,msg) \
  if ( MsgLogger::MsgLogger().logging(MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev)) ) { \
    MsgLogger::MsgLogStream stream(MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev), __FILE__, __LINE__) ;\
    stream.ostream_hack() << msg ; \
  }

#ifdef WithMsgLog
#undef WithMsgLog
#endif
#define WithMsgLog(logger,sev,str) \
  for ( MsgLogger::MsgLogStream str(logger, MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev), __FILE__ , __LINE__) ; str.ok() ; str.finish() )

#ifdef WithMsgLogRoot
#undef WithMsgLogRoot
#endif
#define WithMsgLogRoot(sev,str) \
  for ( MsgLogger::MsgLogStream str(MsgLogger::MsgLogLevel(MsgLogger::MsgLogLevel::sev), __FILE__ , __LINE__) ; str.ok() ; str.finish() )

/**
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

namespace MsgLogger {
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
