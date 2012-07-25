#ifndef RDBMYSQL_QUERYBUF_HH
#define RDBMYSQL_QUERYBUF_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class QueryBuf.
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
class Buffer ;
class Conn ;
}

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RdbMySQL {

/**
 *  This class is a buffer for building SQL queries. It's a bit messy, but
 *  its interface is built for performance, and besides users should not see it.
 *
 *  This software was developed for the BaBar collaboration.  If you
 *  use all or part of it, please give an appropriate acknowledgement.
 *
 *  Copyright (C) 2005 SLAC
 *
 *  @see Query
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class QueryBuf : boost::noncopyable {

public:

  /**
   *  Constructor takes connection object
   *
   *  @param conn   database connection object
   */
  QueryBuf( Conn& conn ) ;

  // Destructor
  virtual ~QueryBuf() ;

  /// resize query buffer to held at list the specified size
  void reserve ( size_t newSize ) ;

  /// return zero-terminated string with query
  const char* str () const { return _query ; }

  /// return current query size
  size_t size() const { return _size ; }

  /// reset query contents
  void clear() { if ( _query ) _query[0] = 0 ; _size = 0 ; }

  // =========== insertion operators for different types ===========

  /// add string, quote and escape depending on the flags
  void append ( const char* str, bool escape = true, bool quote = true ) ;
  void append ( const char* str, size_t size, bool escape = true, bool quote = true ) ;
  void append ( const char* begin, const char* end, bool escape = true, bool quote = true ) ;

  /// add string, quote and escape depending on the flags
  void append ( const std::string& str, bool escape = true, bool quote = true ) ;

  /// add string, quote and escape depending on the flags
  void append ( const Buffer& str, bool escape = true, bool quote = true ) ;

  /// add 1-char string, quote and escape depending on the flags
  void append ( char c, bool escape = true, bool quote = true ) ;

  /// Add TRUE/FALSE
  void append ( bool n, bool escape = true, bool quote = true ) ;

  /// add number, note that escape/quote flags are not used, and present 
  /// only for uniformity (to make templated code happy.) Note that 
  /// signed/unsigned char are formatted as numbers, while char is formatted 
  /// as a 1-char string.
  void append ( signed char c, bool escape = true, bool quote = true ) ;
  void append ( unsigned char c, bool escape = true, bool quote = true ) ;
  void append ( short n, bool escape = true, bool quote = true ) ;
  void append ( unsigned short n, bool escape = true, bool quote = true ) ;
  void append ( int n, bool escape = true, bool quote = true ) ;
  void append ( unsigned int n, bool escape = true, bool quote = true ) ;
  void append ( long n, bool escape = true, bool quote = true ) ;
  void append ( unsigned long n, bool escape = true, bool quote = true ) ;
  void append ( float n, bool escape = true, bool quote = true ) ;
  void append ( double n, bool escape = true, bool quote = true ) ;
  void append ( long double n, bool escape = true, bool quote = true ) ;

protected:

  // Helper functions

private:

  // Friends

  // Data members
  Conn& _conn ;
  char* _query ;
  size_t _size ;
  size_t _capacity ;

};

} // namespace RdbMySQL

#endif // RDBMYSQL_QUERYBUF_HH
