//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Query
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
#include "RdbMySQL/Query.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RdbMySQL/Client.h"
#include "RdbMySQL/Conn.h"
#include "RdbMySQL/Result.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "RdbMySQLQuery";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RdbMySQL {

/**
 *  Constructor takes connection object
 *
 *  @param conn   database connection object
 */
Query::Query ( Conn& conn )
  : _conn(conn)
  , _qbuf(conn)
{
}

// Destructor
Query::~Query ()
{
}

/**
 *  Execute complete query. Different signatures, all about efficiency
 */
Result*
Query::execute ( const char* q )
{
  return this->execute ( q, strlen(q) ) ;
}

Result*
Query::execute ( const std::string& q )
{
  return this->execute ( q.data(), q.size() ) ;
}

Result*
Query::execute ( const char* q, size_t size )
{
  MYSQL* mysql = _conn.mysql() ;
  if ( ! mysql ) {
    MsgLog(logger, error, "Query::execute -- no connection to the database");
    return 0 ;
  }

  Client& client = _conn.client() ;

  // send the query
  if ( client.mysql_real_query ( mysql, q, size ) != 0 ) {
    return 0 ;
  }

  // get the result 
  MYSQL_RES* res = client.mysql_store_result( mysql ) ;
  if ( res ) {
    // return the result
    return new Result( res, &client ) ;
  }

  if ( client.mysql_field_count( mysql ) == 0 ) {
    // it was non-SELECT query, there is no real result, but at least
    // we can tell how many records were changed
    my_ulonglong nrows = client.mysql_affected_rows( mysql ) ;
    return new Result ( nrows ) ;
  }

  // there should be a result but we did not get it which is an error
  return 0 ;
}

} // namespace RdbMySQL
