#ifndef RDBMYSQL_CONN_HH
#define RDBMYSQL_CONN_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Conn.
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

//---------------
// C++ Headers --
//---------------
#include <string>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace RdbMySQL {
class Client ;
}
struct st_mysql ;

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RdbMySQL {

/**
 *  This class represents MySQL connection.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see ConnTable
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Conn : boost::noncopyable {

public:

  /**
   *  Constructor takes parameters which are later passed to mysql_real_connect.
   *
   *  @param host     host name, if empty then localhost is assumed
   *  @param user     user name, if empty then current login name is used
   *  @param passwd   password, a big secuirity hole :)
   *  @param db       database name, if not empty then set default database name
   *  @param port     remote port number to connect to, if 0 then default value is used
   *  @param socket   socket name for local connections
   *  @param client_flag bit mask with varius flags, see mysql_real_connect description
   */
  Conn( const std::string& host,
		const std::string& user, 
		const std::string& passwd, 
		const std::string& db,
		unsigned int port = 0,
		const std::string& socket = std::string(),
		unsigned long client_flag = 0,
		Client* client = 0 ) ;

  /// Destructor, closes connection if it was not closed yet
  ~Conn () ;

  /// open connection
  bool open() ;

  /// close connection
  bool close() ;

  /**
   *  Get the MySQL error code of the last operation, 0 means no error
   */
  unsigned int errnum() const ;

  /**
   *  Get the MySQL error message from the last operation
   */
  const char* error() const ;

  /** 
   *  Get database connection. Returns 0 pointer if not open.
   */
  st_mysql* mysql() ;

  /**
   *  Get a database name
   */
  const std::string& db() const { return _db; }

  /**
   *  Get a pointer to the client interface
   */
  Client& client() const { return *_client ; }

protected:

  // Helper functions

private:

  // Friends

  // Data members
  const std::string _host ;
  const std::string _user ;
  const std::string _passwd ;
  const std::string _db ;
  const unsigned int _port ;
  const std::string _socket ;
  const unsigned long _client_flag ;
  st_mysql* _mysql ;
  Client* _client ;

};

} // namespace RdbMySQL

#endif // RDBMYSQL_CONN_HH
