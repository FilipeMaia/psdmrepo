#ifndef RDBMYSQL_CLIENT_HH
#define RDBMYSQL_CLIENT_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Client.
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
#include <boost/utility.hpp>
#include <mysql/mysql.h>

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

namespace RdbMySQL {

/**
 *  This class represents interface to MySQL client library. The purpose of
 *  the separate interface (instead of using direct mysql_* calls) is that 
 *  it can be implemented with either these direct calls or through the
 *  dynamic loading of the libmysqlclient library.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see ClientTable
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Client : boost::noncopyable {

public:

  /// Destructor, closes connection if it was not closed yet
  virtual ~Client () ;

  /**
   *  Return true if the object represents working interface. I this 
   *  returns false then other methods will not work (will not do anything.)
   *  False returned may mean that shared library failed to load, for example
   *  (but not that connection to server failed.)
   */
  virtual bool working() = 0 ;

  // The rest of the methods are the mirrors of the corresponding mysql_* methods.
  // Only those that are currently needed are present, if you need something 
  // aditional then you have to add it here and implement in specific classes too.

  virtual my_ulonglong mysql_affected_rows(MYSQL *mysql) = 0 ;
  virtual void mysql_close(MYSQL *mysql) = 0 ;
  virtual unsigned int mysql_errno(MYSQL *mysql) = 0 ;
  virtual const char *mysql_error(MYSQL *mysql) = 0 ;
  virtual MYSQL_FIELD *mysql_fetch_field(MYSQL_RES *result) = 0 ;
  virtual MYSQL_FIELD *mysql_fetch_fields(MYSQL_RES *result) = 0 ;
  virtual MYSQL_FIELD *mysql_fetch_field_direct(MYSQL_RES *result, unsigned int fieldnr) = 0 ;
  virtual unsigned long *mysql_fetch_lengths(MYSQL_RES *result) = 0 ;
  virtual MYSQL_ROW mysql_fetch_row(MYSQL_RES *result) = 0 ;
  virtual unsigned int mysql_field_count(MYSQL *mysql) = 0 ;
  virtual void mysql_free_result(MYSQL_RES *result) = 0 ;
  virtual MYSQL *mysql_init(MYSQL *mysql) = 0 ;
  virtual my_ulonglong mysql_insert_id(MYSQL *mysql) = 0 ;
  virtual unsigned int mysql_num_fields(MYSQL_RES *result) = 0 ;
  virtual my_ulonglong mysql_num_rows(MYSQL_RES *result) = 0 ;
  virtual MYSQL *mysql_real_connect(MYSQL *mysql, const char *host, const char *user, 
                                    const char *passwd, const char *db, unsigned int port, 
                                    const char *unix_socket, unsigned long client_flag) = 0 ;
  virtual unsigned long mysql_real_escape_string(MYSQL *mysql, char *to, const char *from, unsigned long length) = 0 ;
  virtual int mysql_real_query(MYSQL *mysql, const char *query, unsigned long length) = 0 ;
  virtual MYSQL_RES *mysql_store_result(MYSQL *mysql) = 0 ;
  virtual MYSQL_RES *mysql_use_result(MYSQL *mysql) = 0 ;

protected:

  /**
   *  Constructor
   */
  Client() {}


private:

};

} // namespace RdbMySQL

#endif // RDBMYSQL_CLIENT_HH
