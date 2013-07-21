//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Conn
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
#include "RdbMySQL/Conn.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <mysql/mysql.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "RdbMySQL/ClientDynamic.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "RdbMySQLConn";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RdbMySQL {

/**
 *  Constructor takes parameters which are later passed to mysql_real_connect.
 *
 *  @param host     host name, if empty then localhost is assumed
 *  @param user     user name, if empty then current login name is used
 *  @param passwd   password, a big secuirity hole :)
 *  @param db       database name, if not empty then set default database name
 *  @param port     remote port number to connect to, if 0 then default value is used
 *  @param socket   socket name for local connections
 */
Conn::Conn ( const std::string& host,
             const std::string& user,
             const std::string& passwd,
             const std::string& db,
             unsigned int port,
             const std::string& socket,
             unsigned long client_flag,
             Client* client )
  : _host(host)
  , _user(user)
  , _passwd(passwd)
  , _db(db)
  , _port(port)
  , _socket(socket)
  , _client_flag(client_flag)
  , _mysql(0)
  , _client(client)
{
  if ( ! _client ) {
    _client = new ClientDynamic() ;
  }
}

/**
 *  @brief Constructor takes connection string.
 *
 *  Format of the connection string:
 *  "Server=hostname;Port=1234;Database=myDataBase;Uid=myUsername;Pwd=myPassword;".
 *  Any part of the connection string can be omitted.
 *
 *  @param connStr  ODBC-like connection string
 *  @param client   if non-zero pointer is passed it will be used instead default
 *                  client instance, will be deleted in destructor
 */
Conn::Conn( const std::string& connStr, Client* client )
  : _host()
  , _user()
  , _passwd()
  , _db()
  , _port()
  , _socket()
  , _client_flag(0)
  , _mysql(0)
  , _client(client)
{
  std::vector<std::string> parts;
  parts.reserve(10);
  boost::split(parts, connStr, boost::is_any_of(";"), boost::token_compress_on);
  for (std::vector<std::string>::const_iterator it = parts.begin(); it != parts.end(); ++ it) {

    std::string part = *it;
    boost::trim(part);
    if (part.empty()) continue;

    std::string::size_type p = part.find('=');
    if (p == std::string::npos) {
      MsgLog(logger, warning, "Conn -- connection string has missing = : " << connStr);
      continue;
    }

    std::string key(part, 0, p);
    boost::to_lower(key);

    const std::string val(part, p+1);

    if (key == "server") {
      _host = val;
    } else if (key == "port") {
      _port = boost::lexical_cast<unsigned>(val);
    } else if (key == "database") {
      _db = val;
    } else if (key == "uid") {
      _user = val;
    } else if (key == "pwd") {
      _passwd = val;
    }

  }

  if ( ! _client ) {
    _client = new ClientDynamic() ;
  }
}


// Destructor
Conn::~Conn ()
{
  close() ;
  delete _client ;
}

/**
 *  open database
 */
bool 
Conn::open()
{
  if ( _mysql ) {
    return true ;
  }
  if ( ! _client->working() ) {
    MsgLog(logger, error, "Conn::open -- mysql client library initialization failed");
    return false ;
  }

  // init structure
  _mysql = _client->mysql_init ( 0 ) ;
  if ( ! _mysql ) {
    MsgLog(logger, error, "Conn::open -- mysql_init failed: " << _client->mysql_error(_mysql));
    return false ;
  }

  // try to connect 
  const char* host = _host.empty() ? 0 : _host.c_str() ;
  const char* user = _user.empty() ? 0 : _user.c_str() ;
  const char* passwd = _passwd.empty() ? 0 : _passwd.c_str() ;
  const char* db = _db.empty() ? 0 : _db.c_str() ;
  const char* socket = _socket.empty() ? 0 : _socket.c_str() ;
  if ( _client->mysql_real_connect ( _mysql, host, user, passwd, db, _port, socket, _client_flag ) == 0 ) {
    MsgLog(logger, error, "Conn::open -- Failed to connect to database: Error: " << _client->mysql_error(_mysql));
    close() ;
    return false ;
  }

  return true ;
}

/**
 *  close database
 */
bool 
Conn::close()
{
  if ( _mysql ) {
    _client->mysql_close ( _mysql ) ;
    _mysql = 0 ;
  }

  return true ;
}

/**
 *  Get the MySQL error code of the last operation, 0 means no error
 */
unsigned int 
Conn::errnum() const
{ 
  return _mysql ? _client->mysql_errno(_mysql) : 0 ; 
}

/**
 *  Get the MySQL error message from the last operation
 */
const char* 
Conn::error() const
{ 
  return _mysql ? _client->mysql_error(_mysql) : "" ; 
}

/** 
 *  Get database connection. Returns 0 pointer if not open.
 */
MYSQL* 
Conn::mysql()
{ 
  return _mysql ; 
}

} // namespace RdbMySQL
