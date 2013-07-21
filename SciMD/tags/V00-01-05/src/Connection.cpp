//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Connection...
//
// Author List:
//      Igor Gaponenko
//
//------------------------------------------------------------------------

#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------

#include "SciMD/Connection.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "SciMD/ConnectionImpl.h"

#include "odbcpp/OdbcEnvironment.h"
#include "odbcpp/OdbcConnection.h"
#include "odbcpp/OdbcException.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

using namespace odbcpp ;

namespace SciMD {

//---------------------------------
// Static members initialization --
//---------------------------------

Connection*
Connection::s_conn = 0 ;

//------------------
// Static methods --
//------------------

Connection*
Connection::open (const std::string& odbc_conn_scimd_str,
                  const std::string& odbc_conn_regdb_str) throw (DatabaseError)
{
    // Disconnect from the previously connected database (if any)
    //
    if ( s_conn != 0 ) {
        delete s_conn ;
        s_conn = 0 ;
    }

    try {

        // Connect to the databases and configure the connector before
        // returning it to a caller.

        // Setup environments
        //
        OdbcEnvironment env_scimd ;
        OdbcEnvironment env_regdb ;

        // Create ODBC connections
        //
        OdbcConnection conn_scimd = env_scimd.connection () ;
        OdbcConnection conn_regdb = env_regdb.connection () ;

        // Set trhose attributes which should be initialized before setting up
        // a connection.
        //
        conn_scimd.setAttr (ODBC_ATTR_PACKET_SIZE(1*1024*1024)) ;
        conn_regdb.setAttr (ODBC_ATTR_PACKET_SIZE(1*1024*1024)) ;

        // Connect now
        //
        conn_scimd.connect (odbc_conn_scimd_str) ;
        conn_regdb.connect (odbc_conn_regdb_str) ;

        // Set connection attributes
        //
        conn_scimd.setAttr (ODBC_AUTOCOMMIT_ON) ;
        conn_regdb.setAttr (ODBC_AUTOCOMMIT_ON) ;

        s_conn = new ConnectionImpl (conn_scimd, conn_regdb) ;

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
    return s_conn ;
}

//----------------
// Constructors --
//----------------

Connection::Connection ()
{}

//--------------
// Destructor --
//--------------

Connection::~Connection () throw ()
{}

} // namespace SciMD
