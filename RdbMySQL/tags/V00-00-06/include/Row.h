#ifndef RDBMYSQL_ROW_HH
#define RDBMYSQL_ROW_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Row.
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

//-------------
// C Headers --
//-------------
extern "C" {
}

//---------------
// C++ Headers --
//---------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RdbMySQL/TypeTraits.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RdbMySQL {

/**
 *  This class represents a single row in the result (relation).
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see Result
 *  @see Header
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Row {

public:

  /**
   *  Constructor takes MYSQL_ROW pointer, and the fields list
   */
  Row ( char** row, unsigned long* lengths ) : _row(row), _lengths(lengths) {}

  // Destructor
  ~Row () {}

  /// get the specified field as a string
  const char* at( unsigned int i ) const { return _row ? _row[i] : 0 ; }

  /// get the size of the string
  unsigned long size( unsigned int i ) const { return _lengths ? _lengths[i] : 0 ; }

  /// do a type conversion
  template <typename T>
  bool at ( unsigned int i, T& val ) const {
    const char* s = at(i) ;
    if ( ! s ) return false ;
    return TypeTraits<T>::str2val ( s, size(i), val ) ;
  }


protected:

  // Helper functions

private:

  // Friends

  // Data members
  char** _row ;
  unsigned long* _lengths ;

};

} // namespace RdbMySQL

#endif // RDBMYSQL_ROW_HH
