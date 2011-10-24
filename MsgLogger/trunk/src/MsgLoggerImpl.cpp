//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MsgLoggerImpl
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
#include "MsgLogger/MsgLoggerImpl.h"

//-------------
// C Headers --
//-------------
extern "C" {
#include <stdlib.h>
}

//---------------
// C++ Headers --
//---------------
#include <algorithm>
#include <map>
#include <sstream>
#include <iostream>
#include <boost/shared_ptr.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgHandler.h"
#include "MsgLogger/MsgHandlerStdStreams.h"
#include "MsgLogger/MsgLogRecord.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // loggers
  typedef std::map< std::string, boost::shared_ptr<MsgLogger::MsgLoggerImpl> > Name2ImplMap ;
  struct Name2Impl {
    Name2Impl () : map(), destroyed(false) {}
    ~Name2Impl () { destroyed = true ; }
    Name2ImplMap map ;
    bool destroyed ;
  } ;
  Name2Impl implementations ;

  // find the parent logger name
  std::string parentName ( const std::string& name ) {
    std::string::size_type n = name.rfind('.') ;
    if ( n == std::string::npos ) {
      return std::string() ;
    } else {
      return std::string( name, 0, n ) ;
    }
  }

  // initialize root logger
  void initRoot ( MsgLogger::MsgLoggerImpl& root ) {
    root.addHandler ( new MsgLogger::MsgHandlerStdStreams ) ;
  }

  // read config stuff
  void readConfig() ;

}

namespace MsgLogger {

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

// Construct named logger
MsgLoggerImpl::MsgLoggerImpl( const std::string& name )
  : _name(name)
  , _level( MsgLogLevel::defaultLevel() )
  , _propagate( ! name.empty() )	      // root logger does not propagate
  , _parent(0)
  , _handlers()
{
}

// Destructor
MsgLoggerImpl::~MsgLoggerImpl()
{
  // destroy all my handlers
  for ( HandlerList::const_iterator it = _handlers.begin() ; it != _handlers.end() ; ++it ) {
    delete *it ;
  }
}


/// add a handler for the messages, takes ownership of the object
void
MsgLoggerImpl::addHandler ( MsgHandler* handler )
{
  _handlers.push_back ( handler ) ;
}


/// check if the specified level will log any message
bool
MsgLoggerImpl::logging ( MsgLogLevel level ) const
{
  // maybe it's time to initialize?
  if ( _name.empty() && _handlers.empty() ) {
    ::initRoot ( const_cast<MsgLoggerImpl&>(*this) ) ;
  }

  // if my own level is higher then don't log through my handlers
  if ( level >= _level ) {
    // if any handler can log then log
    for ( HandlerList::const_iterator it = _handlers.begin() ; it != _handlers.end() ; ++it ) {
      if ( (*it)->logging(level) ) return true ;
    }
  }

  // otherwise ask my parent
  if ( _propagate ) {
    if ( ! _parent ) {
      _parent = getLogger( ::parentName (_name) ) ;
      // if can't make parent then don't try to propagate
      if ( ! _parent ) _propagate = false ;
    }
    if ( _parent ) {
      return _parent->logging(level) ;
    }
  }

  return false ;
}

/// get the stream for the specified log level
bool
MsgLoggerImpl::log ( const MsgLogRecord& record ) const
{
  // maybe it's time to initialize?
  if ( _name.empty() && _handlers.empty() ) {
    ::initRoot ( const_cast<MsgLoggerImpl&>(*this) ) ;
  }

  if ( record.level() >= _level ) {
    // send to all handlers
    for ( HandlerList::const_iterator it = _handlers.begin() ; it != _handlers.end() ; ++it ) {
      (*it)->log(record) ;
    }
  }

  // and send it to my parent
  if ( _propagate ) {
    if ( ! _parent ) {
      _parent = getLogger( ::parentName (_name) ) ;
      // if can't make parent then don't try to propagate
      if ( ! _parent ) _propagate = false ;
    }
    if ( _parent ) {
      _parent->log ( record ) ;
    }
  }

  return true ;
}

MsgLoggerImpl*
MsgLoggerImpl::getLogger( const std::string& name )
{
  if ( implementations.destroyed ) {
    return 0 ;
  }

  // on the first call read configuration
  if ( implementations.map.empty() ) {
    try {
      ::readConfig() ;
    } catch ( const std::exception& e ) {
      std::cerr << "MsgLogger: Error while parsing MSGLOGCONFIG: " << e.what() << std::endl ;
    }
  }

  // maybe we have it already?
  Name2ImplMap::const_iterator it = implementations.map.find ( name ) ;
  if ( it != implementations.map.end() ) {
    return it->second.get() ;
  }

  // need to make a new one
  boost::shared_ptr<MsgLoggerImpl> logger ( new MsgLoggerImpl( name ) ) ;
  implementations.map.insert ( Name2ImplMap::value_type ( name, logger ) ) ;

  return logger.get() ;
}


} // namespace MsgLogger

namespace {

  // make one logger
  void makeLogger ( const std::string& name, const std::string level, bool propagate )
  {
    // check first
    Name2ImplMap::const_iterator it = implementations.map.find ( name ) ;
    if ( it != implementations.map.end() ) {
      return ;
    }

    MsgLogger::MsgLogLevel s ( level ) ;

    boost::shared_ptr<MsgLogger::MsgLoggerImpl> logger ( new MsgLogger::MsgLoggerImpl( name ) ) ;

    // do not setup handler for root logger, it will be done later in initRoot,
    // give user a chance to set different handlers
    if ( ! name.empty() ) logger->addHandler ( new MsgLogger::MsgHandlerStdStreams ) ;

    logger->setLevel ( s ) ;
    logger->propagate ( propagate ) ;
    implementations.map.insert ( Name2ImplMap::value_type ( name, logger ) ) ;

  }

  // read config stuff
  void readConfig()
  {
    const char* env = getenv ( "MSGLOGCONFIG" ) ;
    if ( ! env ) return ;

    const std::string configEnv(env) ;

    // configuration string will have format:
    // level0;logger1,logger2=level1;logger3=level2;...
    // if the level name is given without loggers then it
    // applies to root logger

    typedef std::string::const_iterator iter ;
    for ( iter i = configEnv.begin(), j ; i != configEnv.end() ; i = j ) {

      // find next ;
      j = std::find ( i, configEnv.end(), ';' ) ;

      // find '='
      iter j1 = std::find ( i, j, '=' ) ;
      if ( j1 == j ) {
	// no = means it should be a level name for root logger
	std::string level(i,j) ;
	::makeLogger ( std::string(), level, false ) ;
      } else {
	bool propagate = *(j-1) != '-' ;
	std::string level( j1+1, *(j-1)=='-' ? j-1 : j ) ;

	for ( iter i2 = i, j2 ; i2 != j1 ; i2 = j2 ) {
	  j2 = std::find( i2, j1, ',') ;
	  std::string name ( i2, j2 ) ;
	  if ( j2 != j1 ) ++ j2 ;
	  ::makeLogger ( name, level, propagate ) ;
	}

      }

      // advance past semicolon
      if ( j != configEnv.end() ) ++ j ;

    }

  }


}
