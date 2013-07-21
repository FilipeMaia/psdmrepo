#ifndef RDBMYSQL_RESULT_HH
#define RDBMYSQL_RESULT_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Result.
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
#include "RdbMySQL/Header.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
struct st_mysql_res ;
namespace RdbMySQL {
class Client ;
}

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
 *  @see ResultTable
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Result : boost::noncopyable {

public:

  /**
   *  Constructor takes MYSQL_RES pointer, which must be non-zero
   */
  Result( st_mysql_res* res, Client* client ) ;

  /**
   *  Constructor takes the number of affected rows (for non-SELECT queries)
   */
  Result( unsigned long nrows ) ;

  // Destructor
  ~Result () ;

  /// get the result header. Only makes sense for SELECT-like queries
  Header header() const { return Header(_res,_client) ; }

  /// get the number of the rows in the result.
  unsigned long size() const { return _nrows ; }

protected:

  // Helper functions

private:

  // Friends
  friend class RowIter ;

  // Data members
  st_mysql_res* _res ;
  unsigned long _nrows ;
  Client* _client ;

  // provide access to rows and their lengths
  char** fetch_row() const ;
  unsigned long* fetch_lengths() const ;

};

} // namespace RdbMySQL

#endif // RDBMYSQL_RESULT_HH
