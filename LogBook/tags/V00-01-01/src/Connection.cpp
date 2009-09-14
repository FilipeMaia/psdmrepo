//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: $
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

#include "LogBook/Connection.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <mysql/mysql.h>

#include <iostream>
using namespace std ;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "LogBook/ConnectionImpl.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace LogBook {

//-------------
// Operators --
//-------------

std::ostream&
operator<< (std::ostream& s, const ExperDescr& d)
{
    s << "LogBook::ExperDescr {\n"

      << "         instr_id: " << d.instr_id << "\n"
      << "       instr_name: " << d.instr_name << "\n"
      << "      instr_descr: " << d.instr_descr << "\n"

      << "               id: " << d.id << "\n"
      << "             name: " << d.name << "\n"
      << "            descr: " << d.descr << "\n"

      << "registration_time: " << d.registration_time << "\n"
      << "       begin_time: " << d.begin_time << "\n"
      << "         end_time: " << d.end_time << "\n"

      << "   leader_account: " << d.leader_account << "\n"
      << "     contact_info: " << d.contact_info << "\n"
      << "        posix_gid: " << d.posix_gid << "\n"
      << "}\n" ;
    return s ;
}

std::ostream&
operator<< (std::ostream& s, const ParamInfo& p)
{
    s << "LogBook::ParamInfo {\n"
      << "instrument: " << p.instrument << "\n"
      << "experiment: " << p.experiment << "\n"
      << "      name: " << p.name << "\n"
      << "      type: " << p.type << "\n"
      << "     descr: " << p.descr << "\n"
      << "}\n" ;

    return s ;
}

//---------------------------------
// Static members initialization --
//---------------------------------

Connection*
Connection::s_conn = 0 ;

//------------------
// Static methods --
//------------------

Connection*
Connection::open (const char* host,
                  const char* user,
                  const char* password,
                  const char* logbook,
                  const char* regdb) throw (DatabaseError)
{
    // Disconnect from the previously connected database (if any)
    //
    if ( s_conn != 0 ) {
        delete s_conn ;
        s_conn = 0 ;
    }


    // Prepare the connection object
    //
    MYSQL* mysql = 0;
    if( !(mysql = mysql_init( mysql )))
        throw DatabaseError( "error in mysql_init(): insufficient memory to allocate an object" );

    // Set trhose attributes which should be initialized before setting up
    // a connection.
    //
    ;

    // Connect now
    //
    if( !mysql_real_connect(
        mysql, host, user, password,
        0,  // no default database
        0,  // connect to the default TCP port
        0,  // no defaulkt UNIX socket
        0   // no default client flag
        ))
        throw DatabaseError( std::string( "error in mysql_real_connect(): " ) + mysql_error(mysql));

    // Set connection attributes
    //
    if( mysql_query( mysql, "SET SESSION SQL_MODE='ANSI'" ) ||
        mysql_query( mysql, "SET SESSION AUTOCOMMIT=0" ))
        throw DatabaseError( std::string( "error in mysql_query(): " ) + mysql_error(mysql));

    // Finally, initialize the connector
    //
    ConnectionParams conn_params;
    {
        conn_params.host = ( host == 0 ? "" : host );
        conn_params.user = ( user == 0 ? "" : user );
        conn_params.using_password = !password;
        conn_params.logbook = logbook;
        conn_params.regdb = regdb;
    }
    s_conn = new ConnectionImpl ( mysql, conn_params ) ;

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

} // namespace LogBook
