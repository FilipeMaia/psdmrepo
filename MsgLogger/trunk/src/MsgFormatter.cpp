//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: MsgFormatter.cc,v 1.2 2007/02/11 03:24:25 salnikov Exp $
//
// Description:
//	Class MsgFormatter
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
#include "MsgLogger/MsgFormatter.h"

//-------------
// C Headers --
//-------------
extern "C" {
#include <time.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
}

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogRecord.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  using namespace MsgLogger ;

  // get current time and format it
  void formattedTime ( std::string fmt, std::ostream& out ) ;

  // default format string
  const std::string s_defFmt = "%(time) [%(LVL)] {%(logger)} %(file):%(line) - %(message)" ;

  // these format strings are set with addGlobalFormat
  std::string s_appFmtMap [MsgLogLevel::LAST_LEVEL+1] ;

  // this format strings are set from the environment
  std::string s_envFmtMap [MsgLogLevel::LAST_LEVEL+1] ;

  void initFmtMap() {

    static bool initDone = false ;
    if ( not initDone ) {
      initDone = true ;

      // variable MSGLOGFMT defines global format string
      if ( const char* env = getenv ( "MSGLOGFMT" ) ) {
        std::fill_n ( s_envFmtMap, MsgLogLevel::LAST_LEVEL+1, env ) ;
      }

      // variables MSGLOGFMT_DBG, MSGLOGFMT_TRC, etc. define level-specific
      // format strings
      for ( int i = 0 ; i < MsgLogLevel::LAST_LEVEL+1 ; ++ i ) {
        MsgLogLevel lvl(i) ;
        std::string envname = "MSGLOGFMT_" ;
        envname += lvl.level3() ;
        if ( const char* env = getenv ( envname.c_str() ) ) {
          s_envFmtMap[i] = env ;
        }
      }

    }
  }

}

namespace MsgLogger {

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

// Constructor
MsgFormatter::MsgFormatter( const std::string& afmt, const std::string& timefmt )
  : _timefmt(timefmt)
{
  // initialize global format map
  ::initFmtMap() ;

  if ( _timefmt.empty() ) {
    if ( const char* env = getenv ( "MSGLOGTIMEFMT" ) ) {
      _timefmt = env ;
    } else {
      _timefmt = "%Y-%m-%d %H:%M:%S.%f" ;
    }
  }
}

// Destructor
MsgFormatter::~MsgFormatter()
{
}

// set format for all formatters
void
MsgFormatter::addGlobalFormat ( const std::string& fmt )
{
  std::fill_n ( ::s_appFmtMap, MsgLogLevel::LAST_LEVEL+1, fmt ) ;
}

void
MsgFormatter::addGlobalFormat ( MsgLogLevel level, const std::string& fmt )
{
  ::s_appFmtMap[level.code()] = fmt ;
}

// add level-specific format
void
MsgFormatter::addFormat ( MsgLogLevel level, const std::string& fmt )
{
  _fmtMap[level.code()] = fmt ;
}

// format message to the output stream
void
MsgFormatter::format ( const MsgLogRecord& rec, std::ostream& out )
{
  const std::string& fmt = getFormat( rec.level() );

  // read format and fill the stream
  for ( std::string::const_iterator i = fmt.begin() ; i != fmt.end() ; ++ i ) {

    if ( *i != '%' ) {
      out.put( *i ) ;
      continue ;
    }

    std::string::const_iterator j = i ;
    if ( ++j == fmt.end() ) {
      out.put( *i ) ;
      continue ;
    }

    // escaped percent
    if ( *j == '%' ) {
      out.put( '%' ) ;
      i = j ;
      continue ;
    }

    // should be opening paren after percent
    if ( *j != '(' ) {
      out.put( *i ) ;
      continue ;
    }

    // find closing paren
    j = std::find ( j, fmt.end(), ')' ) ;
    if ( j == fmt.end() ) {
      out.put( *i ) ;
      continue ;
    }

    // get the name between parens
    std::string name ( i+2, j ) ;
    bool known = true ;
    if ( name == "logger" ) {
      if ( rec.logger().empty() ) {
        out << "/root/" ;
      } else {
        out << rec.logger() ;
      }
    } else if ( name == "level" ) {
      out << rec.level() ;
    } else if ( name == "L" ) {
      out << rec.level().levelLetter() ;
    } else if ( name == "LVL" ) {
      out << rec.level().level3() ;
    } else if ( name == "message" ) {
      // hack - reset buffer state
      rec.msgbuf()->pubseekoff ( 0, std::ios_base::beg, std::ios_base::in ) ;
      out << rec.msgbuf() ;
    } else if ( name == "path" ) {
      const char* path = rec.fileName() ;
      out << ( path ? path : "<empty>" ) ;
    } else if ( name == "file" ) {
      const char* path = rec.fileName() ;
      if ( path ) {
	const char* p = strrchr ( path, '/' ) ;
	if ( ! p ) p = path ;
	out << p+1 ;
      } else {
	out << "<empty>" ;
      }
    } else if ( name == "line" ) {
      out << rec.lineNum() ;
    } else if ( name == "time" ) {
      formattedTime ( _timefmt, out ) ;
    } else if ( name == "pid" ) {
      out << (unsigned long)getpid() ;
    } else {
      known = false ;
    }

    // advance
    if ( known ) {
      i = j ;
    }

  }


}

// get a format string for a given level
const std::string&
MsgFormatter::getFormat ( MsgLogLevel level ) const
{
  int lvl = level.code() ;
  if ( not _fmtMap[lvl].empty() ) {
    return _fmtMap[lvl] ;
  } else if ( not ::s_envFmtMap[lvl].empty() ) {
    return ::s_envFmtMap[lvl] ;
  } else if ( not ::s_appFmtMap[lvl].empty() ) {
    return ::s_appFmtMap[lvl] ;
  } else {
    return ::s_defFmt ;
  }
}



} // namespace MsgLogger

namespace {

  // get current time and format it
  void formattedTime ( std::string fmt, std::ostream& out )
  {
    // get seconds/nanoseconds
    struct timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );

    // convert to break-down time
    struct tm tms ;
    localtime_r( &ts.tv_sec, &tms );

    // replace %f in the format string with miliseconds
    std::string::size_type n = fmt.find("%f") ;
    if ( n != std::string::npos ) {
      char subs[4] ;
      snprintf ( subs, 4, "%03d", int(ts.tv_nsec/1000000) ) ;
      while ( n != std::string::npos ) {
	fmt.replace ( n, 2, subs ) ;
	n = fmt.find("%f") ;
      }
    }

    char buf[1024] ;
    strftime(buf, 1024, fmt.c_str(), &tms );
    out << buf ;

  }

}
