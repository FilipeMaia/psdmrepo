//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class QueryBuf
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
#include "RdbMySQL/QueryBuf.h"

//---------------
// C++ Headers --
//---------------
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RdbMySQL/Buffer.h"
#include "RdbMySQL/Client.h"
#include "RdbMySQL/Conn.h"
#include "RdbMySQL/TypeTraits.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  size_t initialBufSize = 512 ;

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
QueryBuf::QueryBuf( Conn& conn )
  : _conn(conn)
  , _query(0)
  , _size(0)
  , _capacity(0)
{
}

// Destructor
QueryBuf::~QueryBuf ()
{
  delete [] _query ;
}

/// add string, quote and escape depending on the flags
void
QueryBuf::append ( const char* str, bool escape, bool quote )
{
  if ( str ) {
    append ( str, str+strlen(str), escape, quote ) ;
  } else {
    append ( "NULL", false, false ) ;
  }
}

void 
QueryBuf::append ( const char* str, size_t size, bool escape, bool quote )
{
  if ( str ) {
    append ( str, str+size, escape, quote ) ;
  } else {
    append ( "NULL", false, false ) ;
  }
}

void 
QueryBuf::append ( const char* begin, const char* end, bool escape, bool quote )
{
  size_t strSize = end - begin ;
  size_t maxSize = strSize ;
  if ( escape ) maxSize *= 2 ;
  if ( quote ) maxSize += 2 ;

  size_t newMaxSize = _size + maxSize ;
  // need one spare char for \0
  if ( newMaxSize >= _capacity ) reserve ( newMaxSize ) ;
  assert ( _capacity > newMaxSize ) ;

  if ( quote ) _query[_size++] = '\'' ;

  if ( escape ) {
    assert ( _conn.mysql() ) ;
    _size += _conn.client().mysql_real_escape_string ( _conn.mysql(), _query+_size, begin, strSize ) ;
  } else {
    std::copy ( begin, end, _query+_size ) ;
    _size += strSize ;
    _query[_size] = '\0' ;
  }

  if ( quote ) {
    _query[_size++] = '\'' ;
    _query[_size] = '\0' ;
  }
}

void
QueryBuf::append ( const std::string& str, bool escape, bool quote )
{
  if ( ! str.empty() ) {
    append ( str.data(), str.data()+str.size(), escape, quote ) ;
  } else {
    append ( "", escape, quote ) ;
  }
}

void
QueryBuf::append ( const Buffer& str, bool escape, bool quote )
{
  if ( str.data() ) {
    append ( str.data(), str.data()+str.size(), escape, quote ) ;
  } else {
    append ( "NULL", false, false ) ;
  }
}

/// add 1-char string, quote and escape depending on the flags, mysql only needed
/// if escape is true
void 
QueryBuf::append ( char c, bool escape, bool quote )
{
  append ( &c, &c+1, escape, quote ) ;
}

/// Add TRUE/FALSE
void 
QueryBuf::append ( bool n, bool, bool )
{
  append ( TypeTraits<bool>::val2str(n), false, false ) ;
}

/// add number

void 
QueryBuf::append ( signed char c, bool, bool )
{
  append ( TypeTraits<signed char>::val2str(c), false, false ) ;
}

void 
QueryBuf::append ( unsigned char c, bool, bool )
{
  append ( TypeTraits<unsigned char>::val2str(c), false, false ) ;
}

void 
QueryBuf::append ( short n, bool, bool )
{
  append ( TypeTraits<short>::val2str(n), false, false ) ;
}

void 
QueryBuf::append ( unsigned short n, bool, bool )
{
  append ( TypeTraits<unsigned short>::val2str(n), false, false ) ;
}

void 
QueryBuf::append ( int n, bool, bool )
{
  append ( TypeTraits<int>::val2str(n), false, false ) ;
}

void 
QueryBuf::append ( unsigned int n, bool, bool )
{
  append ( TypeTraits<unsigned int>::val2str(n), false, false ) ;
}

void 
QueryBuf::append ( long n, bool, bool )
{
  append ( TypeTraits<long>::val2str(n), false, false ) ;
}

void 
QueryBuf::append ( unsigned long n, bool, bool )
{
  append ( TypeTraits<unsigned long>::val2str(n), false, false ) ;
}

void 
QueryBuf::append ( float n, bool, bool )
{
  append ( TypeTraits<float>::val2str(n), false, false ) ;
}

void 
QueryBuf::append ( double n, bool, bool )
{
  append ( TypeTraits<double>::val2str(n), false, false ) ;
}

void 
QueryBuf::append ( long double n, bool, bool )
{
  append ( TypeTraits<long double>::val2str(n), false, false ) ;
}

/// resize query buffer to held at list the specified size
void 
QueryBuf::reserve ( size_t newSize )
{
  // note we need 1 additional byte for the terminating 0
  size_t newCapacity = ( newSize / ::initialBufSize + 1 ) * ::initialBufSize ;
  if ( newCapacity <= _capacity ) {
    return ;
  }

  char* newBuf = new char[newCapacity] ;
  if ( _query ) {
    std::copy ( _query, _query+_size+1, newBuf ) ;
    delete [] _query ;
  } else {
    newBuf[0] = '\0' ;
  }
  _query = newBuf ;
  _capacity = newCapacity ;
}

} // namespace RdbMySQL
