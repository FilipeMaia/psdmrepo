//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RdbMySQLDumpTable...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

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
#include "RdbMySQL/Query.h"
#include "RdbMySQL/Result.h"
#include "RdbMySQL/Row.h"
#include "RdbMySQL/RowIter.h"
#include "RdbMySQL/Conn.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace std ;

namespace {

  template <typename Iter>
  void dashes ( Iter begin, Iter end ) {
    cout.fill('-') ;
    for ( ; begin != end ; ++ begin ) {
      cout << "+" << setw(*begin+2) << "" ;
    }
    cout << "+\n" ;
    cout.fill(' ') ;
  }

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace RdbMySQL {

//
//  Application class
//
class RdbMySQLDumpTable : public AppUtils::AppBase {
public:

  // Constructor
  explicit RdbMySQLDumpTable ( const std::string& appName ) ;

  // destructor
  ~RdbMySQLDumpTable () ;

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

private:

  // more command line options and arguments
  AppUtils::AppCmdOpt<std::string>  m_optHost ;
  AppUtils::AppCmdOpt<unsigned int> m_optPort ;
  AppUtils::AppCmdOpt<std::string>  m_optUser ;
  AppUtils::AppCmdArg<std::string>  m_argDb ;
  AppUtils::AppCmdArg<std::string>  m_argTable ;

  // other data members
  int m_val ;

};

//----------------
// Constructors --
//----------------
RdbMySQLDumpTable::RdbMySQLDumpTable ( const std::string& appName )
  : AppUtils::AppBase( appName )
  , m_optHost ( 's', "host", "string", "remote host name", "" )
  , m_optPort ( 'p', "port", "number", "remote port number", 0 )
  , m_optUser ( 'u', "user", "string", "user name", "" )
  , m_argDb ( "database", "database name" )
  , m_argTable ( "table", "table name" )
{
  addOption( m_optHost );
  addOption( m_optPort );
  addOption( m_optUser );
  addArgument( m_argDb );
  addArgument( m_argTable );
}

//--------------
// Destructor --
//--------------
RdbMySQLDumpTable::~RdbMySQLDumpTable ()
{
}

/**
 *  Main method which runs the whole application
 */
int
RdbMySQLDumpTable::runApp ()
{

  // make a conection object
  Conn conn( m_optHost.value(), m_optUser.value(), "", m_argDb.value(), m_optPort.value() ) ;

  // open connection
  if ( ! conn.open() ) {
    cerr << "Cannot open connection" << endl ;
    return 1 ;
  }

  // make a query object
  Query query( conn ) ;

  const char* q = "SELECT * FROM ?" ;
  std::auto_ptr<Result> res ( query.executePar( q, m_argTable.value() ) ) ;
  if ( ! res.get() ) {
    MsgLogRoot(error, "query failed: " << conn.error());
    return 2 ;
  }

  // result header
  const Header& header = res->header() ;
  int nf = header.size() ;
  if ( ! nf ) {
    MsgLogRoot(error, "empty header");
    return 3 ;
  }

  // calculate max field sizes
  std::vector<int> sizes( nf, 0 ) ;
  for ( int i = 0 ; i < nf ; ++ i ) {
    const Field& field = header.field(i) ;
    sizes[i] = strlen(field.name()) ;
    if ( sizes[i] < field.max_length() ) {
      sizes[i] = field.max_length() ;
    }
  }

  // print header
  cout.setf( ios::left, ios::adjustfield ) ;
  ::dashes ( sizes.begin(), sizes.end() ) ;
  for ( int i = 0 ; i < nf ; ++ i ) {
    const Field& field = header.field(i) ;
    cout << "| " << setw(sizes[i]) << field.name() << " " ;
  }
  cout << "|\n" ;
  ::dashes ( sizes.begin(), sizes.end() ) ;

  // print all rows
  RowIter iter ( *res ) ;
  while ( iter.next() ) {
    Row row = iter.row() ;
    for ( int i = 0 ; i < nf ; ++ i ) {
      const char* str = row.at(i) ;
      if ( ! str ) str = "NULL" ;
      cout << "| " << setw(sizes[i]) << str << " " ;
    }
    cout << "|\n" ;
  }

  ::dashes ( sizes.begin(), sizes.end() ) ;

  return 0 ;
}

} // namespace RdbMySQL


// this defines main()
APPUTILS_MAIN(RdbMySQL::RdbMySQLDumpTable)
