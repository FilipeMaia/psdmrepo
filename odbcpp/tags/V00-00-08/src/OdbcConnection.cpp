//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcConnection...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "odbcpp/OdbcConnection.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcException.h"
#include "odbcpp/OdbcLog.h"
#include "odbcpp/OdbcStatement.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  using namespace odbcpp ;

  // the "deleter" operator which disconnect from the database
  struct Disconnecter {
    void operator()( OdbcHandle<OdbcConn>* h ) {
      // do a disconnect
      /*SQLRETURN status = */SQLDisconnect ( *(*h) ) ;
      //OdbcStatusCheckMsg ( status, *h, "OdbcConnection: failed to disconnect from database" ) ;

      // now free the handle itself
      delete h ;
    }
  };

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace odbcpp {

  //----------------
// Constructors --
//----------------
OdbcConnection::OdbcConnection ( OdbcHandle<OdbcEnv> envH, OdbcHandle<OdbcConn> connH )
  : m_envH(envH)
  , m_connH( new OdbcHandle<OdbcConn>(connH), Disconnecter() )
  , m_connString()
{
}

//--------------
// Destructor --
//--------------
OdbcConnection::~OdbcConnection ()
{
}

// Connect to the database or throw exception
// Format of the connection string is the same as for SQLDriverConnect
void
OdbcConnection::connect( const std::string& connString )
{
  // open the connection
  SQLCHAR connStrRet[256];
  SQLSMALLINT connStrSize ;
  SQLRETURN status = SQLDriverConnect ( *(*m_connH), 0,
                              (SQLCHAR*)connString.c_str(), connString.size(),
                              connStrRet, sizeof connStrRet, &connStrSize,
                              SQL_DRIVER_NOPROMPT ) ;
  OdbcStatusCheckMsg ( status, *m_connH, "OdbcConnection::connect -- failed to connect to database, check connection string" ) ;

  m_connString = std::string( (char*)connStrRet, connStrSize ) ;
  OdbcLog ( debug, "OdbcConnection: successfully connected to " << m_connString ) ;
}

// get the satement objects
OdbcStatement
OdbcConnection::statement( const std::string& q )
{
  OdbcHandle<OdbcStmt> stmtH = OdbcHandle<OdbcStmt>::make( *m_connH ) ;
  OdbcHandleCheck( stmtH, *m_connH ) ;

  SQLRETURN status = SQLPrepare ( *stmtH, (SQLCHAR*)q.data(), q.size() ) ;
  OdbcStatusCheck ( status, stmtH );

  return OdbcStatement ( stmtH ) ;
}

// get the statement objects for the list of the tables
OdbcResultPtr
OdbcConnection::tables( const std::string& catPattern,
                        const std::string& schemaPattern,
                        const std::string& tblNamePattern,
                        const std::string& tblTypePattern )
{
  OdbcHandle<OdbcStmt> stmtH = OdbcHandle<OdbcStmt>::make( *m_connH ) ;
  OdbcHandleCheck( stmtH, *m_connH ) ;

  SQLRETURN status = SQLTables ( *stmtH,
              (SQLCHAR*)catPattern.data(), catPattern.size(),
              (SQLCHAR*)schemaPattern.data(), schemaPattern.size(),
              (SQLCHAR*)tblNamePattern.data(), tblNamePattern.size(),
              (SQLCHAR*)tblTypePattern.data(), tblTypePattern.size() ) ;
  OdbcStatusCheck ( status, stmtH );

  return OdbcResultPtr ( new OdbcResult ( stmtH ) ) ;}

} // namespace odbcpp