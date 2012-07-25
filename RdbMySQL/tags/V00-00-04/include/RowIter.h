#ifndef RDBMYSQL_ROWITER_HH
#define RDBMYSQL_ROWITER_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RowIter.
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
class Result ;
class Row ;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RdbMySQL {

/**
 *  This class is an iterator for the rows in the result set.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see Result
 *  @see Row
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class RowIter : boost::noncopyable {

public:

  /**
   *  Constructor takes MYSQL_ROW pointer, and the fields list
   */
  RowIter( const Result& res ) : _res(res), _row(0) {}

  // Destructor
  ~RowIter () {}

  /// advance, return true if there is a row to extract
  bool next() ;

  /// get the current row
  Row row() const ;

protected:

  // Helper functions

private:

  // Friends

  // Data members
  const Result& _res ;
  char** _row ;                 // that's MYSQL_ROW actually

};

} // namespace RdbMySQL

#endif // RDBMYSQL_ROWITER_HH
