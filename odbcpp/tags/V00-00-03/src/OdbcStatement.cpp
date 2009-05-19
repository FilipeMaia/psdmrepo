//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcStatement...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "odbcpp/OdbcStatement.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "unixodbc/sql.h"
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
OdbcStatement::OdbcStatement (OdbcHandle<OdbcStmt> stmtH)
  : m_stmtH ( stmtH )
{
}

//--------------
// Destructor --
//--------------
OdbcStatement::~OdbcStatement ()
{
}

// unbind all parameters
void
OdbcStatement::unbindParams ()
{
  SQLRETURN r = SQLFreeStmt ( *m_stmtH, SQL_RESET_PARAMS ) ;
  OdbcStatusCheck ( r, m_stmtH );
}


// execute statement
OdbcResultPtr
OdbcStatement::execute()
{
  SQLRETURN status = SQLExecute ( *m_stmtH ) ;
  OdbcStatusCheck ( status, m_stmtH );

  return OdbcResultPtr ( new OdbcResult ( m_stmtH ) ) ;
}

} // namespace odbcpp
