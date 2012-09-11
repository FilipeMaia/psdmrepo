//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ClientDynamic
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
#include "RdbMySQL/ClientDynamic.h"

//---------------
// C++ Headers --
//---------------
#include <stdlib.h>
#include <dlfcn.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "mysql/libmysql_soname.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RdbMySQL {

/**
 *  Constructor
 */
ClientDynamic::ClientDynamic()
  : Client()
  , _libh(0)
  , _triedOpen(false)
{
}

// Destructor
ClientDynamic::~ClientDynamic ()
{
}

/**
 *  Return true if the object represents working interface. I this 
 *  returns false then other methods will not work (will not do anything.)
 *  False returned may mean that shared library failed to load, for example
 *  (but not that connection to server failed.)
 */
bool 
ClientDynamic::working()
{
  if ( ! _triedOpen ) {
    this->openLib() ;
  }
  return _libh != 0 ;
}

// The rest of the methods are the mirrors of the corresponding mysql_* methods.
// Only those that are currently needed are present, if you need something 
// aditional then you have to add it here and implement in specific classes too.

my_ulonglong 
ClientDynamic::mysql_affected_rows(MYSQL *mysql)
{
  typedef my_ulonglong (*Func)(MYSQL *mysql) ;
  static Func fun = (Func)findMethod( "mysql_affected_rows" ) ;

  if ( fun == 0 ) {
    return my_ulonglong() ;
  }
  return (*fun)(mysql) ;
}

void 
ClientDynamic::mysql_close(MYSQL *mysql)
{
  typedef void (*Func)(MYSQL *mysql) ;
  static Func fun = (Func)findMethod( "mysql_close" ) ;

  if ( fun == 0 ) {
    return ;
  }
  return (*fun)(mysql) ;
}

unsigned int 
ClientDynamic::mysql_errno(MYSQL *mysql)
{
  typedef unsigned int (*Func)(MYSQL *mysql);
  static Func fun = (Func)findMethod( "mysql_errno" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(mysql) ;
}

const char *
ClientDynamic::mysql_error(MYSQL *mysql)
{
  typedef const char *(*Func)(MYSQL *mysql);
  static Func fun = (Func)findMethod( "mysql_error" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(mysql) ;
}

MYSQL_FIELD *
ClientDynamic::mysql_fetch_field(MYSQL_RES *result)
{
  typedef MYSQL_FIELD *(*Func)(MYSQL_RES *result);
  static Func fun = (Func)findMethod( "mysql_fetch_field" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(result) ;
}

MYSQL_FIELD *
ClientDynamic::mysql_fetch_fields(MYSQL_RES *result)
{
  typedef MYSQL_FIELD *(*Func)(MYSQL_RES *result);
  static Func fun = (Func)findMethod( "mysql_fetch_fields" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(result) ;
}

MYSQL_FIELD *
ClientDynamic::mysql_fetch_field_direct(MYSQL_RES *result, unsigned int fieldnr)
{
  typedef MYSQL_FIELD *(*Func)(MYSQL_RES *result, unsigned int fieldnr);
  static Func fun = (Func)findMethod( "mysql_fetch_field_direct" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(result,fieldnr) ;
}

unsigned long *
ClientDynamic::mysql_fetch_lengths(MYSQL_RES *result)
{
  typedef unsigned long *(*Func)(MYSQL_RES *result);
  static Func fun = (Func)findMethod( "mysql_fetch_lengths" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(result) ;
}

MYSQL_ROW 
ClientDynamic::mysql_fetch_row(MYSQL_RES *result)
{
  typedef MYSQL_ROW (*Func)(MYSQL_RES *result) ;
  static Func fun = (Func)findMethod( "mysql_fetch_row" ) ;

  if ( fun == 0 ) {
    return MYSQL_ROW(0) ;
  }
  return (*fun)(result) ;
}

unsigned int 
ClientDynamic::mysql_field_count(MYSQL *mysql)
{
  typedef unsigned int (*Func)(MYSQL *mysql) ;
  static Func fun = (Func)findMethod( "mysql_field_count" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(mysql) ;
}

void 
ClientDynamic::mysql_free_result(MYSQL_RES *result)
{
  typedef void (*Func)(MYSQL_RES *result) ;
  static Func fun = (Func)findMethod( "mysql_free_result" ) ;

  if ( fun == 0 ) {
    return ;
  }
  return (*fun)(result) ;
}

MYSQL *
ClientDynamic::mysql_init(MYSQL *mysql)
{
  typedef MYSQL *(*Func)(MYSQL *mysql);
  static Func fun = (Func)findMethod( "mysql_init" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(mysql) ;
}

my_ulonglong 
ClientDynamic::mysql_insert_id(MYSQL *mysql)
{
  typedef my_ulonglong (*Func)(MYSQL *mysql);
  static Func fun = (Func)findMethod( "mysql_insert_id" ) ;

  if ( fun == 0 ) {
    return my_ulonglong() ;
  }
  return (*fun)(mysql) ;
}

unsigned int 
ClientDynamic::mysql_num_fields(MYSQL_RES *result)
{
  typedef unsigned int (*Func)(MYSQL_RES *result);
  static Func fun = (Func)findMethod( "mysql_num_fields" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(result) ;
}

my_ulonglong 
ClientDynamic::mysql_num_rows(MYSQL_RES *result)
{
  typedef my_ulonglong (*Func)(MYSQL_RES *result);
  static Func fun = (Func)findMethod( "mysql_num_rows" ) ;

  if ( fun == 0 ) {
    return my_ulonglong() ;
  }
  return (*fun)(result) ;
}

MYSQL *
ClientDynamic::mysql_real_connect(MYSQL *mysql, const char *host, const char *user,
				  const char *passwd, const char *db, unsigned int port, 
				  const char *unix_socket, unsigned long client_flag)
{
  typedef MYSQL *(*Func)(MYSQL *mysql, const char *host, const char *user,
				    const char *passwd, const char *db, unsigned int port, 
				    const char *unix_socket, unsigned long client_flag) ;
  static Func fun = (Func)findMethod( "mysql_real_connect" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(mysql,host,user,passwd,db,port,unix_socket,client_flag) ;
}

unsigned long 
ClientDynamic::mysql_real_escape_string(MYSQL *mysql, char *to, const char *from, unsigned long length)
{
  typedef unsigned long (*Func)(MYSQL *mysql, char *to, const char *from, unsigned long length);
  static Func fun = (Func)findMethod( "mysql_real_escape_string" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(mysql,to,from,length) ;
}

int 
ClientDynamic::mysql_real_query(MYSQL *mysql, const char *query, unsigned long length)
{
  typedef int (*Func)(MYSQL *mysql, const char *query, unsigned long length);
  static Func fun = (Func)findMethod( "mysql_real_query" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(mysql,query,length) ;
}

MYSQL_RES *
ClientDynamic::mysql_store_result(MYSQL *mysql)
{
  typedef MYSQL_RES *(*Func)(MYSQL *mysql) ;
  static Func fun = (Func)findMethod( "mysql_store_result" ) ;

  if ( fun == 0 ) {
    return my_ulonglong() ;
  }
  return (*fun)(mysql) ;
}

MYSQL_RES *
ClientDynamic::mysql_use_result(MYSQL *mysql)
{
  typedef MYSQL_RES *(*Func)(MYSQL *mysql);
  static Func fun = (Func)findMethod( "mysql_use_result" ) ;

  if ( fun == 0 ) {
    return 0 ;
  }
  return (*fun)(mysql) ;
}


// open the shared library, if successful it will set _libh to non-zero
void 
ClientDynamic::openLib()
{
  if ( _triedOpen ) return ;
  _triedOpen = true ;

  _libh = dlopen ( libmysql_soname, RTLD_LAZY | RTLD_GLOBAL ) ;
}

// find the method in the shared library
void* 
ClientDynamic::findMethod ( const char* name )
{
  // may need to open the shared lib first
  if ( ! _triedOpen ) {
    this->openLib() ;
  }

  // if was not found then can't find anything
  if ( _libh == 0 ) {
    return 0 ;
  }

  // look for it
   return dlsym ( _libh, name ) ;
}

} // namespace RdbMySQL
