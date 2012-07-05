#ifndef RDBMYSQL_QUERY_HH
#define RDBMYSQL_QUERY_HH

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Query.
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
#include "RdbMySQL/QueryBuf.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace RdbMySQL {
class Result ;
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
 *  @see QueryTable
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Query : boost::noncopyable {

public:

  /**
   *  Constructor takes connection object
   *
   *  @param conn   database connection object
   */
  Query ( Conn& conn ) ;

  // Destructor
  virtual ~Query () ;

  /**
   *  Execute complete query. Different signatures, all for efficiency.
   */
  virtual Result* execute ( const char* q ) ;
  virtual Result* execute ( const std::string& q ) ;
  virtual Result* execute ( const char* q, size_t size ) ;

  /**
   *  Execute query with parameters. Given the static nature of C++ this is the 
   *  best that I can do (without other over-complicated stuff). If you need more
   *  parameters just add more methods here or compose queries yourself.
   *  Returns the result of the query or zero pointer for failed queries.
   *  User is responsible for deleting returned object.
   */
  template <typename T1>
  Result* executePar ( const std::string& q, const T1& p1 ) ;
  
  template <typename T1, typename T2>
  Result* executePar ( const std::string& q, const T1& p1, const T2& p2 ) ;
  
  template <typename T1, typename T2, typename T3>
  Result* executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3 ) ;
  
  template <typename T1, typename T2, typename T3, typename T4>
  Result* executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
    const T4& p4 ) ;
  
  template <typename T1, typename T2, typename T3, typename T4, typename T5>
  Result* executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
    const T4& p4, const T5& p5 ) ;
  
  template <typename T1, typename T2, typename T3, typename T4, typename T5, 
            typename T6>
  Result* executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
    const T4& p4, const T5& p5, const T6& p6 ) ;
  
  template <typename T1, typename T2, typename T3, typename T4, typename T5, 
            typename T6, typename T7>
  Result* executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
    const T4& p4, const T5& p5, const T6& p6, const T7& p7 ) ;

  template <typename T1, typename T2, typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8>
  Result* executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
    const T4& p4, const T5& p5, const T6& p6, const T7& p7, const T8& p8 ) ;

  template <typename T1, typename T2, typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, typename T9>
  Result* executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
    const T4& p4, const T5& p5, const T6& p6, const T7& p7, const T8& p8, const T9& p9 ) ;

  template <typename T1, typename T2, typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, typename T9, typename T10>
  Result* executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
    const T4& p4, const T5& p5, const T6& p6, const T7& p7, const T8& p8, const T9& p9, const T10& p10 ) ;

  /// return the query string from last executePar() (not execute())
  const char* str() const { return _qbuf.str() ; }

protected:

  // Helper functions

private:

  // Friends

  // Data members
  Conn& _conn ;
  QueryBuf _qbuf ;

  // Adds part of the query before ? and the value in place of the ?. Returns
  // the index of the char following ?.
  template <typename T1>
  std::string::size_type 
  appendToQuery ( const std::string& q, std::string::size_type n, const T1& p1 ) ;

};


template <typename T1>
Result*
Query::executePar ( const std::string& q, const T1& p1 )
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

template <typename T1, typename T2>
Result*
Query::executePar ( const std::string& q, const T1& p1, const T2& p2 )
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p2 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

template <typename T1, typename T2, typename T3>
Result*
Query::executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3 )
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p2 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p3 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

template <typename T1, typename T2, typename T3, typename T4>
Result*
Query::executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
		         const T4& p4 ) 
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p2 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p3 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p4 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
Result*
Query::executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
		         const T4& p4, const T5& p5 ) 
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p2 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p3 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p4 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p5 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
Result*
Query::executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
		         const T4& p4, const T5& p5, const T6& p6 ) 
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p2 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p3 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p4 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p5 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p6 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
Result*
Query::executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
		         const T4& p4, const T5& p5, const T6& p6, const T7& p7 ) 
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p2 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p3 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p4 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p5 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p6 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p7 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, 
	  typename T7, typename T8>
Result*
Query::executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
		         const T4& p4, const T5& p5, const T6& p6, const T7& p7,
		         const T8& p8 ) 
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p2 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p3 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p4 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p5 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p6 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p7 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p8 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, 
	  typename T7, typename T8, typename T9>
Result*
Query::executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
		         const T4& p4, const T5& p5, const T6& p6, const T7& p7,
		         const T8& p8, const T9& p9 ) 
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p2 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p3 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p4 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p5 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p6 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p7 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p8 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p9 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, 
	  typename T7, typename T8, typename T9, typename T10>
Result*
Query::executePar ( const std::string& q, const T1& p1, const T2& p2, const T3& p3,
		         const T4& p4, const T5& p5, const T6& p6, const T7& p7,
		         const T8& p8, const T9& p9, const T10& p10 ) 
{
  _qbuf.clear() ;
  std::string::size_type n = 0 ;
  if ( ( n = appendToQuery( q, n, p1 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p2 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p3 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p4 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p5 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p6 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p7 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p8 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p9 ) ) == std::string::npos ) return false ;
  if ( ( n = appendToQuery( q, n, p10 ) ) == std::string::npos ) return false ;
  _qbuf.append( q.data()+n, q.data()+q.size(), false, false ) ; // add rest of the query

  return execute ( _qbuf.str(), _qbuf.size() ) ;
}

// Adds part of the query before ? and the value in place of the ?. Returns
// the index of the char following ?.
template <typename T1>
std::string::size_type 
Query::appendToQuery ( const std::string& q, std::string::size_type n, const T1& p1 )
{
  std::string::size_type n1 = q.find('?',n) ;   // find next ?
  if ( n1 == std::string::npos ) return n1 ;
  _qbuf.append( q.data()+n, q.data()+n1, false, false ) ; // add query up to ?, do not escape
  ++ n1 ;
  if ( n1 < q.size() && q[n1] == '?' ) {
    // two ?? means quote/escape strings, for non-strings it does not make difference
    _qbuf.append( p1, true, true ) ;		// add a parameter, escape and quote
    ++ n1 ;
  } else {
    _qbuf.append( p1, false, false ) ;	// add a parameter, do not escape
  }
  return n1 ;
}

} // namespace RdbMySQL

#endif // RDBMYSQL_QUERY_HH
