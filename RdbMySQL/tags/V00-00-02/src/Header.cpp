//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Row
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
#include "RdbMySQL/Header.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <string.h>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RdbMySQL/Client.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------


namespace RdbMySQL {

///  Returns the numbr of fields in a row
unsigned int 
Header::size() const
{ 
  return _res ? _client->mysql_num_fields(_res) : 0 ; 
}

/// get the field at nth position
Field
Header::field( unsigned int i ) const
{ 
  return Field( _client->mysql_fetch_field_direct(_res,i) ) ;
}

/// get the index of the field from its name, returns -1 if no such field exist
int 
Header::index ( const char* name ) const
{
  int n = _client->mysql_num_fields(_res) ;
  MYSQL_FIELD* fields = _client->mysql_fetch_fields(_res) ;
  for ( int i = 0 ; i < n ; ++ i ) {
    if ( strcmp ( name, fields[i].name ) == 0 ) {
      return i ;
    }
  }
  return -1 ;
}

int 
Header::index ( const std::string& name ) const
{
  int n = _client->mysql_num_fields(_res) ;
  MYSQL_FIELD* fields = _client->mysql_fetch_fields(_res) ;
  for ( int i = 0 ; i < n ; ++ i ) {
    if ( name == fields[i].name ) {
      return i ;
    }
  }
  return -1 ;
}

} // namespace RdbMySQL
