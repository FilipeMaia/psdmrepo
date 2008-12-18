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

namespace {

  template <typename T>
  void printColVal ( const char* pfx, const OdbcColumnVar<T>& col, int row )
  {
    cout << pfx ;
    if ( col.isNull(row) ) {
      cout << "<NULL>" ;
    } else {
      cout << col.value(row) ;
    }
  }

}

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

  // print complete list of tables
  void listTables ( OdbcConnection conn ) ;

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

  listTables ( conn ) ;

  // create statement
  OdbcStatement stmt = conn.statement( "SELECT * FROM r_coll_main WHERE coll_id > ? AND coll_name LIKE ?" ) ;
  OdbcParam<int> coll_id(10003) ;
  OdbcParam<std::string> coll_name("%/home%") ;
  stmt.bindParam ( 1, coll_id ) ;
  stmt.bindParam ( 2, coll_name ) ;

  // execute a query
  OdbcResultPtr result = stmt.execute() ;
  stmt.unbindParams() ;

  cout << "result.empty(): " << ( result->empty() ? "yes" : "no" ) << '\n' ;
  const OdbcResultHeader& header = result->header() ;
  for ( OdbcResultHeader::const_iterator i = header.begin() ; i != header.end() ; ++ i ) {
    cout << "  column " << i->columnNumber() << ": " << i->columnName() << '\n' ;
  }

  const unsigned int nRows = 4 ;

  OdbcColumnVar<int> col1(nRows) ;
  OdbcColumnVar<std::string> col2(256,nRows) ;
  OdbcColumnVar<std::string> col3(256,nRows) ;
  OdbcColumnVar<std::string> col4(256,nRows) ;
  OdbcColumnVar<std::string> col5(256,nRows) ;
  OdbcColumnVar<int> col6(nRows) ;
  result->bindColumn( 1, col1 ) ;
  result->bindColumn( 2, col2 ) ;
  result->bindColumn( 3, col3 ) ;
  result->bindColumn( "coll_owner_name", col4 ) ;
  result->bindColumn( "coll_owner_zone", col5 ) ;
  result->bindColumn( "coll_map_id", col6 ) ;

  int r = 1 ;
  unsigned int fetched ;
  while ( ( fetched = result->fetch() ) > 0 ) {
    cout << " fetched " << fetched << " rows\n";
    for ( unsigned int i = 0 ; i < fetched ; ++ i ) {
      cout << " row " << r << " :: " ;
      printColVal ( "  1:", col1, i ) ;
      printColVal ( "  2:", col2, i ) ;
      printColVal ( "  3:", col3, i ) ;
      printColVal ( "  4:", col4, i ) ;
      printColVal ( "  5:", col5, i ) ;
      printColVal ( "  6:", col6, i ) ;
      cout << '\n' ;
      ++ r ;
    }
  }

  // try to fetch the same result again
  SQLSMALLINT orient = SQL_FETCH_FIRST ;
  r = 1 ;
  while ( ( fetched = result->fetchScroll( orient, 0 ) ) > 0 ) {
    cout << " fetched " << fetched << " rows\n";
    for ( unsigned int i = 0 ; i < fetched ; ++ i ) {
      cout << " row " << r << " :: " ;
      printColVal ( "  1:", col1, i ) ;
      printColVal ( "  2:", col2, i ) ;
      printColVal ( "  3:", col3, i ) ;
      printColVal ( "  4:", col4, i ) ;
      printColVal ( "  5:", col5, i ) ;
      printColVal ( "  6:", col6, i ) ;
      cout << '\n' ;
      ++ r ;
      orient = SQL_FETCH_NEXT ;
    }
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

// print complete list of tables
void
OdbcSimpleTest::listTables ( OdbcConnection conn )
{
  OdbcResultPtr result = conn.tables( "%", "%", "%", "" ) ;

  cout << "result.empty(): " << ( result->empty() ? "yes" : "no" ) << '\n' ;
  if ( result->empty() ) return ;

  const unsigned int nRows = 100 ;

  OdbcColumnVar<std::string> col1(256,nRows) ;
  OdbcColumnVar<std::string> col2(256,nRows) ;
  OdbcColumnVar<std::string> col3(256,nRows) ;
  OdbcColumnVar<std::string> col4(256,nRows) ;
  OdbcColumnVar<std::string> col5(256,nRows) ;
  result->bindColumn( 1, col1 ) ;
  result->bindColumn( 2, col2 ) ;
  result->bindColumn( 3, col3 ) ;
  result->bindColumn( 4, col4 ) ;
  result->bindColumn( 5, col5 ) ;

  cout << " ========== List of tables ==========\n" ;
  const OdbcResultHeader& header = result->header() ;
  for ( OdbcResultHeader::const_iterator i = header.begin() ; i != header.end() ; ++ i ) {
    cout << "  column " << i->columnNumber() << ": " << i->columnName() << '\n' ;
  }

  unsigned int fetched ;
  while ( ( fetched = result->fetch() ) > 0 ) {
    for ( unsigned int i = 0 ; i < fetched ; ++ i ) {
      printColVal ( "  1:", col1, i ) ;
      printColVal ( "  2:", col2, i ) ;
      printColVal ( "  3:", col3, i ) ;
      printColVal ( "  4:", col4, i ) ;
      printColVal ( "  5:", col5, i ) ;
      cout << '\n' ;
    }
  }

}


// this defines main()
APPUTILS_MAIN(OdbcSimpleTest)
