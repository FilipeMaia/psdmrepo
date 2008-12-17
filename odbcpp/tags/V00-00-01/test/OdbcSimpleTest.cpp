//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcSimpleTest...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "Lusi/Lusi.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdArg.h"
#include "AppUtils/AppCmdOpt.h"
#include "MsgLogger/MsgLogger.h"
#include "odbcpp/OdbcEnvironment.h"
#include "odbcpp/OdbcConnection.h"
#include "odbcpp/OdbcResult.h"
#include "odbcpp/OdbcStatement.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

using namespace std ;
using namespace odbcpp ;

//
//  Application class
//
class OdbcSimpleTest : public AppUtils::AppBase {
public:

  // Constructor
  explicit OdbcSimpleTest ( const std::string& appName ) ;

  // destructor
  ~OdbcSimpleTest () ;

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

private:

  // more command line options and arguments
  AppUtils::AppCmdArg<std::string> m_connStr ;

};

//----------------
// Constructors --
//----------------
OdbcSimpleTest::OdbcSimpleTest ( const std::string& appName )
  : AppUtils::AppBase( appName )
  , m_connStr( "conn-string", "ODBC connection string" )
{
  addArgument( m_connStr ) ;
}

//--------------
// Destructor --
//--------------
OdbcSimpleTest::~OdbcSimpleTest ()
{
}

int
OdbcSimpleTest::runApp ()
try {

  // setup environment
  OdbcEnvironment env ;

  // print the list of drivers
  typedef OdbcEnvironment::DriverList DList ;
  DList drivers = env.drivers() ;
  cout << "Drivers:\n";
  for ( DList::const_iterator i = drivers.begin() ; i != drivers.end() ; ++ i ) {
    cout << "   " << *i  <<  '\n';
  }

  // print the list of data sources
  typedef OdbcEnvironment::DsList DsList ;
  DsList sources = env.dataSources() ;
  cout << "Data sources:\n";
  for ( DsList::const_iterator i = sources.begin() ; i != sources.end() ; ++ i ) {
    cout << "   " << *i  <<  '\n';
  }

  // get few attributes
  cout << "env[ODBC_ATTR_CONNECTION_POOLING] = " << env.getAttr<ODBC_ATTR_CONNECTION_POOLING>() << '\n' ;
  cout << "env[ODBC_ATTR_CP_MATCH] = " << env.getAttr<ODBC_ATTR_CP_MATCH>() << '\n' ;
  cout << "env[ODBC_ATTR_ODBC_VERSION] = " << env.getAttr<ODBC_ATTR_ODBC_VERSION>() << '\n' ;
  cout << "env[ODBC_ATTR_OUTPUT_NTS] = " << env.getAttr<ODBC_ATTR_OUTPUT_NTS>() << '\n' ;

  // create a connection
  OdbcConnection conn = env.connection() ;

  // set few attributes
  conn.setAttr ( ODBC_ATTR_PACKET_SIZE(1*1024*1024) ) ;

  // connect now
  conn.connect ( m_connStr.value() ) ;

  // get few attributes
  cout << "con[ODBC_ATTR_ACCESS_MODE] = " << conn.getAttr<ODBC_ATTR_ACCESS_MODE>() << '\n' ;
  //cout << "con[ODBC_ATTR_ASYNC_ENABLE] = " << conn.getAttr<ODBC_ATTR_ASYNC_ENABLE>() << '\n' ;
  //cout << "con[ODBC_ATTR_AUTO_IPD] = " << conn.getAttr<ODBC_ATTR_AUTO_IPD>() << '\n' ;
  cout << "con[ODBC_ATTR_AUTOCOMMIT] = " << conn.getAttr<ODBC_ATTR_AUTOCOMMIT>() << '\n' ;
  //cout << "con[ODBC_ATTR_CONNECTION_DEAD] = " << conn.getAttr<ODBC_ATTR_CONNECTION_DEAD>() << '\n' ;
  //cout << "con[ODBC_ATTR_CONNECTION_TIMEOUT] = " << conn.getAttr<ODBC_ATTR_CONNECTION_TIMEOUT>() << '\n' ;
  cout << "con[ODBC_ATTR_CURRENT_CATALOG] = " << conn.getAttr<ODBC_ATTR_CURRENT_CATALOG>() << '\n' ;
  cout << "con[ODBC_ATTR_LOGIN_TIMEOUT] = " << conn.getAttr<ODBC_ATTR_LOGIN_TIMEOUT>() << '\n' ;
  cout << "con[ODBC_ATTR_PACKET_SIZE] = " << conn.getAttr<ODBC_ATTR_PACKET_SIZE>() << '\n' ;

  // create statement
  OdbcStatement stmt = conn.statement( "SELECT * FROM r_coll_main WHERE coll_id > ?" ) ;
  OdbcParam<int> coll_id(1000) ;
  stmt.bindParam ( 1, coll_id ) ;

  // execute a query
  OdbcResultPtr result = stmt.execute() ;
  stmt.unbindParams() ;

  cout << "result.empty(): " << ( result->empty() ? "yes" : "no" ) << '\n' ;
  const OdbcResultHeader& header = result->header() ;
  for ( OdbcResultHeader::const_iterator i = header.begin() ; i != header.end() ; ++ i ) {
    cout << "  column " << i->columnNumber() << ": " << i->columnName() << '\n' ;
  }

  OdbcColumnVar<int> col1 ;
  OdbcColumnVar<std::string> col2(256) ;
  OdbcColumnVar<std::string> col3(256) ;
  OdbcColumnVar<std::string> col4(256) ;
  OdbcColumnVar<std::string> col5(256) ;
  OdbcColumnVar<int> col6 ;
  result->bindColumn( 1, col1 ) ;
  result->bindColumn( 2, col2 ) ;
  result->bindColumn( 3, col3 ) ;
  result->bindColumn( "coll_owner_name", col4 ) ;
  result->bindColumn( "coll_owner_zone", col5 ) ;
  result->bindColumn( "coll_map_id", col6 ) ;

  while ( result->fetch() ) {
    cout << "  1:" << col1.value() ;
    cout << "  2:" << col2.value() ;
    cout << "  3:" << col3.value() ;
    cout << "  4:" << col4.value() ;
    cout << "  5:" << col5.value() ;
    cout << "  6:" << col6.value() ;
    cout << '\n' ;
  }

  // return 0 on success, other values for error (like main())
  return 0 ;

} catch ( const std::exception& e ) {
  MsgLogRoot(error,"exception caught: " << e.what() ) ;
  return 2;
} catch(...) {
  MsgLogRoot(error,"unknown exception caught" ) ;
  return 2;
}


// this defines main()
APPUTILS_MAIN(OdbcSimpleTest)
