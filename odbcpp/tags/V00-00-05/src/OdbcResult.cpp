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
  , m_fetched()
  , m_rowArraySize(0)
{
  // set the pointer to location where number of rows will be set by fetch
  SQLRETURN r = SQLSetStmtAttr ( *m_stmtH, SQL_ATTR_ROWS_FETCHED_PTR, SQLPOINTER(&m_fetched), 0 ) ;
  OdbcStatusCheck ( r, m_stmtH ) ;
}

//--------------
// Destructor --
//--------------
OdbcResult::~OdbcResult ()
{
  closeCursor() ;
  unbindColumns() ;
  SQLSetStmtAttr ( *m_stmtH, SQL_ATTR_ROWS_FETCHED_PTR, SQLPOINTER(0), 0 ) ;
}

// Set the number of rows returned in a single fetch operation
// All OdbcColumnVar should use the same size then. Note that
// this sets statement attribute, be careful because the same
// statement can be used by different results. Destructor resets
// this value back to 1.
void
OdbcResult::setRowArraySize( unsigned int rowArraySize )
{
  if ( m_rowArraySize > 0 and rowArraySize > m_rowArraySize ) {
    OdbcExceptionThrow("OdbcResult::setRowArraySize - requested size is too large") ;
  }
  SQLRETURN r = SQLSetStmtAttr ( *m_stmtH, SQL_ATTR_ROW_ARRAY_SIZE, SQLPOINTER(SQLUINTEGER(rowArraySize)), 0 ) ;
  OdbcStatusCheck ( r, m_stmtH ) ;
  m_rowArraySize = rowArraySize ;
}

// unbind all bound columns
void
OdbcResult::unbindColumns()
{
  SQLRETURN r = SQLFreeStmt ( *m_stmtH, SQL_UNBIND ) ;
  OdbcStatusCheck ( r, m_stmtH );

  m_rowArraySize = 0 ;
  setRowArraySize( 1 ) ;
}

// fetch the next set of rows, return number of rows fetched
unsigned int
OdbcResult::fetch()
{
  SQLRETURN r = SQLFetch ( *m_stmtH );
  if ( r == SQL_NO_DATA ) return 0 ;
  OdbcStatusCheck ( r, m_stmtH ) ;
  return m_fetched ;
}

// fetch the row at offset, return number of rows fetched.
// Parameters are the same as for SQLFetchScroll
unsigned int
OdbcResult::fetchScroll( int orientation, int offset )
{
  SQLRETURN r = SQLFetchScroll ( *m_stmtH, orientation, offset );
  if ( r == SQL_NO_DATA ) return 0 ;
  OdbcStatusCheck ( r, m_stmtH ) ;
  return m_fetched ;
}

// close the cursor and discard all pending data
void
OdbcResult::closeCursor()
{
  SQLRETURN r = SQLFreeStmt ( *m_stmtH, SQL_CLOSE ) ;
  OdbcStatusCheck ( r, m_stmtH );
}

void
OdbcResult::updateRowArraySize( unsigned int nRow )
{
  // For the first bound column set the array size equal to the size
  // of the bound variable. Later bound variables can reduce the array
  // size but never increase it.
  if ( m_rowArraySize == 0 or nRow < m_rowArraySize ) {
    setRowArraySize( nRow ) ;
  }
}

} // namespace odbcpp
