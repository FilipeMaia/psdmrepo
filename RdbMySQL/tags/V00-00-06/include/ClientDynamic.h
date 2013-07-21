#ifndef RDBMYSQL_CLIENTDYNAMIC_HH
#define RDBMYSQL_CLIENTDYNAMIC_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ClientDynamic.
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

//----------------------
// Base Class Headers --
//----------------------
#include "RdbMySQL/Client.h"

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
 *  This is an implementation of the Client interface using the dynamic
 *  loading of the libmysqlclient shared library.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see ClientDynamicTable
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ClientDynamic : public Client {

public:

  /**
   *  Constructor
   */
  ClientDynamic() ;


  /// Destructor, closes connection if it was not closed yet
  virtual ~ClientDynamic () ;

  /**
   *  Return true if the object represents working interface. I this 
   *  returns false then other methods will not work (will not do anything.)
   *  False returned may mean that shared library failed to load, for example
   *  (but not that connection to server failed.)
   */
  virtual bool working() ;

  // The rest of the methods are the mirrors of the corresponding mysql_* methods.
  // Only those that are currently needed are present, if you need something 
  // aditional then you have to add it here and implement in specific classes too.

  virtual my_ulonglong mysql_affected_rows(MYSQL *mysql) ;
  virtual void mysql_close(MYSQL *mysql) ;
  virtual unsigned int mysql_errno(MYSQL *mysql) ;
  virtual const char *mysql_error(MYSQL *mysql) ;
  virtual MYSQL_FIELD *mysql_fetch_field(MYSQL_RES *result) ;
  virtual MYSQL_FIELD *mysql_fetch_fields(MYSQL_RES *result) ;
  virtual MYSQL_FIELD *mysql_fetch_field_direct(MYSQL_RES *result, unsigned int fieldnr) ;
  virtual unsigned long *mysql_fetch_lengths(MYSQL_RES *result) ;
  virtual MYSQL_ROW mysql_fetch_row(MYSQL_RES *result) ;
  virtual unsigned int mysql_field_count(MYSQL *mysql) ;
  virtual void mysql_free_result(MYSQL_RES *result) ;
  virtual MYSQL *mysql_init(MYSQL *mysql) ;
  virtual my_ulonglong mysql_insert_id(MYSQL *mysql) ;
  virtual unsigned int mysql_num_fields(MYSQL_RES *result) ;
  virtual my_ulonglong mysql_num_rows(MYSQL_RES *result) ;
  virtual MYSQL *mysql_real_connect(MYSQL *mysql, const char *host, const char *user, 
                                    const char *passwd, const char *db, unsigned int port, 
                                    const char *unix_socket, unsigned long client_flag) ;
  virtual unsigned long mysql_real_escape_string(MYSQL *mysql, char *to, const char *from, unsigned long length) ;
  virtual int mysql_real_query(MYSQL *mysql, const char *query, unsigned long length) ;
  virtual MYSQL_RES *mysql_store_result(MYSQL *mysql) ;
  virtual MYSQL_RES *mysql_use_result(MYSQL *mysql) ;

protected:


private:

  // Friends

  // Data members
  void* _libh ;
  bool  _triedOpen ;

  // open the shared library, if successful it will set _libh to non-zero
  void openLib() ;

  // find the method in the shared library
  void* findMethod ( const char* name ) ;

};

} // namespace RdbMySQL

#endif // RDBMYSQL_CLIENTDYNAMIC_HH
