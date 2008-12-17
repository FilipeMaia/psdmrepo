//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcResult...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "odbcpp/OdbcResult.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace odbcpp {

//----------------
// Constructors --
//----------------
OdbcResult::OdbcResult ( OdbcHandle<OdbcStmt> stmtH )
  : m_stmtH( stmtH )
  , m_header ( stmtH )
{
}

//--------------
// Destructor --
//--------------
OdbcResult::~OdbcResult ()
{
  closeCursor() ;
  unbindColumns() ;
}

// unbind all bound columns
void
OdbcResult::unbindColumns()
{
  SQLRETURN r = SQLFreeStmt ( *m_stmtH, SQL_UNBIND ) ;
  OdbcStatusCheck ( r, m_stmtH );
}

// fetch the next row
bool
OdbcResult::fetch()
{
  SQLRETURN r = SQLFetch ( *m_stmtH );
  if ( r == SQL_NO_DATA ) return false ;
  OdbcStatusCheck ( r, m_stmtH ) ;
  return true ;
}

// fetch the row at offset, parameters are the same as for SQLFetchScroll
bool
OdbcResult::fetchScroll( int orientation, int offset )
{
  SQLRETURN r = SQLFetchScroll ( *m_stmtH, orientation, offset );
  if ( r == SQL_NO_DATA ) return false ;
  OdbcStatusCheck ( r, m_stmtH ) ;
  return true ;
}

// close the cursor and discard all pending data
void
OdbcResult::closeCursor()
{
  SQLRETURN r = SQLFreeStmt ( *m_stmtH, SQL_CLOSE ) ;
  OdbcStatusCheck ( r, m_stmtH );
}

} // namespace odbcpp
