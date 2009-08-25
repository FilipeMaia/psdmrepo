//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcResultHeader...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "odbcpp/OdbcResultHeader.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcException.h"

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
OdbcResultHeader::OdbcResultHeader ( OdbcHandle<OdbcStmt> stmtH )
  : m_columns()
{
  // get the number of columns
  SQLSMALLINT columnCount ;
  SQLRETURN r = SQLNumResultCols( *stmtH, &columnCount );
  OdbcStatusCheck ( r, stmtH ) ;

  // reserve space for all columns
  m_columns.reserve( columnCount ) ;
  for ( SQLSMALLINT i = 1 ; i <= columnCount ; ++ i ) {
    m_columns.push_back ( OdbcColumnDescr( stmtH, i ) ) ;
  }
}

//--------------
// Destructor --
//--------------
OdbcResultHeader::~OdbcResultHeader ()
{
}

// get the column index for given column name, returns negative number for
// non-existing column names
const OdbcColumnDescr*
OdbcResultHeader::findColumn( const std::string& columnName ) const
{
  for ( const_iterator i = begin() ; i != end() ; ++ i ) {
    if ( i->columnName() == columnName ) return &(*i) ;
  }
  return 0 ;
}

} // namespace odbcpp
