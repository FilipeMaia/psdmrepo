#ifndef MSGLOGLEVEL_HH
#define MSGLOGLEVEL_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: MsgLogLevel.h,v 1.2 2005/07/26 18:09:14 salnikov Exp $
//
// Description:
//	Class MsgLogLevel.
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

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  This class defines message logging levels, their names and ordering.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see MsgLogLevelMsgLogLevel
 *
 *  @version $Id: MsgLogLevel.h,v 1.2 2005/07/26 18:09:14 salnikov Exp $
 *
 *  @author Andy Salnikov
 */

namespace MsgLogger {
class MsgLogLevel {

public:

  /**
   *  Message Level levels.
   */
  enum Level { debug,
	       trace,
	       info,
	       warning,
	       error,
	       nolog      // There should be no messages with this level, it's only for loggers
             };

  // Construct root logger
  MsgLogLevel( Level code ) : _level(code) {}

  // default copy ctor is OK
  //MsgLogLevel( const MsgLogLevel& );

  // Destructor
  ~MsgLogLevel() {}

  // default assignment is OK
  //MsgLogLevel& operator= ( const MsgLogLevel& );
  MsgLogLevel& operator= ( const Level code ) { _level = code ; return *this ;}

  // comparison operators
  bool operator == ( MsgLogLevel other ) const { return _level == other._level ; }
  bool operator != ( MsgLogLevel other ) const { return _level != other._level ; }
  bool operator < ( MsgLogLevel other ) const { return int(_level) < int(other._level) ; }
  bool operator <= ( MsgLogLevel other ) const { return int(_level) <= int(other._level) ; }
  bool operator > ( MsgLogLevel other ) const { return int(_level) > int(other._level) ; }
  bool operator >= ( MsgLogLevel other ) const { return int(_level) >= int(other._level) ; }

  // get full printable name of Level level
  const char* levelName () const ;

  // get 3-letter printable name of Level code
  const char* level3 () const ;

  // get one-char Level code
  char levelLetter () const ;

protected:

  // Helper functions

private:

  // Data
  Level _level ;

//------------------
// Static Members --
//------------------

public:

  // default level for loggers
  static MsgLogLevel defaultLevel () { return MsgLogLevel(info) ; }

};

inline std::ostream&
operator<< ( std::ostream& o, MsgLogLevel sev ) {
  return o << sev.levelName() ;
}

} // namespace MsgLogger

#endif // MSGLOGLEVEL_HH
