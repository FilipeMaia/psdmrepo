//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Result
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
#include "RdbMySQL/Result.h"

//---------------
// C++ Headers --
//---------------
#include <mysql/mysql.h>
#include <string>
#include <assert.h>

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

/**
 *  Constructor takes MYSQL_RES pointer, which must be non-zero
 */
Result::Result( MYSQL_RES* res, Client* client )
  : _res(res)
  , _nrows()
  , _client(client)
{
  assert ( _res ) ;
  assert ( _client ) ;
  _nrows = _client->mysql_num_rows(_res) ;
}

/**
 *  Constructor takes the number of affected rows (for non-SELECT queries)
 */
Result::Result( unsigned long nrows )
  : _res(0)
  , _nrows(nrows)
  , _client(0)
{
}


// Destructor
Result::~Result ()
{
  if ( _res ) _client->mysql_free_result ( _res ) ;
}

// provide access to rows and their lengths
MYSQL_ROW 
Result::fetch_row() const
{
  if ( ! _res ) return 0 ;
  return _client->mysql_fetch_row(_res) ;
}

unsigned long* 
Result::fetch_lengths() const
{
  if ( ! _res ) return 0 ;
  return _client->mysql_fetch_lengths(_res) ;
}

} // namespace RdbMySQL
