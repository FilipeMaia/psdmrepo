//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcEnvironment...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "odbcpp/OdbcEnvironment.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcConnection.h"
#include "odbcpp/OdbcException.h"
#include "odbcpp/OdbcLog.h"
#include "unixodbc/sqlext.h"

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
OdbcEnvironment::OdbcEnvironment ()
  : m_envH( OdbcHandle<OdbcEnv>::make(OdbcHandle<OdbcEnv>()) )
{
  OdbcHandleCheck ( m_envH, "OdbcEnvironment -- failed to create OdbcEnv handle" ) ;
  OdbcLog ( debug, "OdbcEnvironment: created OdbcEnv handle " << m_envH ) ;

  // have to declare the application ODBC version, use v.3 now
  setAttr ( ODBC_OV_ODBC3 ) ;
}

//--------------
// Destructor --
//--------------
OdbcEnvironment::~OdbcEnvironment ()
{
}

// get the list of drivers
OdbcEnvironment::DriverList
OdbcEnvironment::drivers()
{
  DriverList result ;

  for ( SQLUSMALLINT dir = SQL_FETCH_FIRST ; ; dir = SQL_FETCH_NEXT ) {
    SQLCHAR descr[128] ;
    SQLCHAR attr[512] ;
    SQLSMALLINT descrLen, attrLen ;

    //OdbcLog( debug, "OdbcEnvironment::drivers" ) ;
    SQLRETURN r = SQLDrivers( *m_envH, dir, descr, sizeof descr, &descrLen, attr, sizeof attr, &attrLen ) ;
    if ( SQL_SUCCEEDED(r) ) {
      //OdbcLog( debug, "OdbcEnvironment::drivers: " << descrLen << " " << descr ) ;
      std::string driver((char*)descr,descrLen) ;
      std::replace ( attr, attr+attrLen, '\0', ';' ) ;
      std::string attrib((char*)attr,attrLen) ;
      result.push_back ( OdbcDriverDescription ( driver, attrib ) ) ;
    } else {
      break ;
    }
  }
  return result ;
}

// get the list of data sources
OdbcEnvironment::DsList
OdbcEnvironment::dataSources()
{
  DsList result ;

  for ( SQLUSMALLINT dir = SQL_FETCH_FIRST ; ; dir = SQL_FETCH_NEXT ) {
    SQLCHAR ds[128] ;
    SQLCHAR drv[128] ;
    SQLSMALLINT dsLen, drvLen ;

    SQLRETURN r = SQLDataSources( *m_envH, dir, ds, sizeof ds, &dsLen, drv, sizeof drv, &drvLen ) ;
    if ( SQL_SUCCEEDED(r) ) {
      std::string dsName((char*)ds,dsLen) ;
      std::string driver((char*)drv,drvLen) ;
      result.push_back ( OdbcDataSource ( dsName, driver ) ) ;
    } else {
      break ;
    }
  }
  return result ;
}

// Connect to the database or throw exception
// Format of the connection string is the same as for SQLDriverConnect
OdbcConnection
OdbcEnvironment::connection()
{
  // make connection handle
  OdbcHandle<OdbcConn> connH = OdbcHandle<OdbcConn>::make( m_envH ) ;
  OdbcHandleCheckMsg ( connH, m_envH, "OdbcEnvironment::connect -- failed to create OdbcConn handle" ) ;
  OdbcLog ( debug, "OdbcEnvironment: created OdbcConn handle " << connH ) ;

  return OdbcConnection( m_envH, connH ) ;
}


} // namespace odbcpp
