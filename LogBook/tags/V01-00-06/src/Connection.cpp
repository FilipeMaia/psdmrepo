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

//-----------------------
// This Class's Header --
//-----------------------

#include "LogBook/Connection.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <mysql/mysql.h>

#include <iostream>
#include <fstream>
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
// Functions --
//-------------

/**
  * Parse the next line of a configuration file.
  *
  * The lines are expected to have the following format:
  *
  *   <key>=[<value>]
  *
  * The function will also check that the specified key is the one found
  * in the line.
  *
  * @return 'true' and set a value of the parameter, or 'false' otherwise
  */
bool
parse_next_line (std::string& value, const char* key, std::ifstream& str)
{
    std::string line;
    if( str >> line ) {
        const size_t separator_pos = line.find( '=' );
        if( separator_pos != std::string::npos ) {
            const std::string key = line.substr( 0, separator_pos );
            if( key == line.substr( 0, separator_pos )) {
                value = line.substr( separator_pos + 1 );
                return true;
            }
        }
    }
    return false;
}

/**
  * Return a pointinr onto a C-style string or null pointer if the input string is empty
  */
inline
const char*
string_or_null (const std::string& str)
{
    if( str.empty()) return 0 ;
    return str.c_str();
}

/**
  * Establish database connection for the specified parameters
  */
MYSQL*
connect2server (const std::string& host,
                const std::string& user,
                const std::string& password,
                const std::string& db) throw (DatabaseError)
{
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
        mysql,
        string_or_null( host ),
        string_or_null( user ),
        string_or_null( password ),
        string_or_null( db ),
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

    return mysql;
}

//-------------
// Operators --
//-------------

std::ostream&
operator<< (std::ostream& s, const ExperDescr& d)
{
    s << "LogBook::ExperDescr {\n"

      << "           instr_id: " << d.instr_id << "\n"
      << "         instr_name: " << d.instr_name << "\n"
      << "        instr_descr: " << d.instr_descr << "\n"

      << "                 id: " << d.id << "\n"
      << "               name: " << d.name << "\n"
      << "              descr: " << d.descr << "\n"

      << "    registration_time: " << d.registration_time << "\n"
      << "         begin_time: " << d.begin_time << "\n"
      << "           end_time: " << d.end_time << "\n"

      << "     leader_account: " << d.leader_account << "\n"
      << "       contact_info: " << d.contact_info << "\n"
      << "          posix_gid: " << d.posix_gid << "\n"
      << "}\n" ;
    return s ;
}

std::ostream&
operator<< (std::ostream& s, const ParamInfo& p)
{
    s << "LogBook::ParamInfo {\n"
      << "  instrument: " << p.instrument << "\n"
      << "  experiment: " << p.experiment << "\n"
      << "        name: " << p.name << "\n"
      << "        type: " << p.type << "\n"
      << "       descr: " << p.descr << "\n"
      << "}\n" ;

    return s ;
}

std::ostream&
operator<< (std::ostream& s, const AttrInfo& a)
{
    s << "LogBook::AttrInfo {\n"
      << "  instrument: " << a.instrument << "\n"
      << "  experiment: " << a.experiment << "\n"
      << "         run: " << a.run << "\n"
      << "  attr_class: " << a.attr_class << "\n"
      << "   attr_name: " << a.attr_name << "\n"
      << "   attr_type: " << a.attr_type << "\n"
      << "  attr_descr: " << a.attr_descr << "\n"
      << "}\n" ;

    return s ;
}


//------------------
// Static methods --
//------------------

Connection*
Connection::open ( const std::string& config) throw (WrongParams,
                                                     DatabaseError)
{
    if( config.empty())
        throw WrongParams( "error in Connection::open(): found 0 pointer instead of configuration file" );

    // Read configuration parameters and decript the passwords.
    //
    ifstream config_file( config.c_str());
    if( !config_file.good())
        throw WrongParams( "error in Connection::open(): failed to open the configuration file: '"+config+"'" );

    std::string logbook_host;
    std::string logbook_user;
    std::string logbook_password;
    std::string logbook_db;

    std::string regdb_host;
    std::string regdb_user;
    std::string regdb_password;
    std::string regdb_db;

    std::string ifacedb_host;
    std::string ifacedb_user;
    std::string ifacedb_password;
    std::string ifacedb_db;

    if( !( parse_next_line( logbook_host,     "logbook_host",     config_file ) &&
           parse_next_line( logbook_user,     "logbook_user",     config_file ) &&
           parse_next_line( logbook_password, "logbook_password", config_file ) &&
           parse_next_line( logbook_db,       "logbook_db",       config_file ) &&

           parse_next_line( regdb_host,       "regdb_host",       config_file ) &&
           parse_next_line( regdb_user,       "regdb_user",       config_file ) &&
           parse_next_line( regdb_password,   "regdb_password",   config_file ) &&
           parse_next_line( regdb_db,         "regdb_db",         config_file ) &&

           parse_next_line( ifacedb_host,     "ifacedb_host",     config_file ) &&
           parse_next_line( ifacedb_user,     "ifacedb_user",     config_file ) &&
           parse_next_line( ifacedb_password, "ifacedb_password", config_file ) &&
           parse_next_line( ifacedb_db,       "ifacedb_db",       config_file )))

         throw WrongParams( "error in Connection::open(): failed to parse the configuration file: '"+config+"'" );

    // Open and configure connections, and create the API object.
    //
    return new ConnectionImpl (
        connect2server ( logbook_host, logbook_user, logbook_password, logbook_db ),
        connect2server (   regdb_host,   regdb_user,   regdb_password,   regdb_db ),
        connect2server ( ifacedb_host, ifacedb_user, ifacedb_password, ifacedb_db )) ;
}

Connection*
Connection::open ( const std::string& logbook_host,
                   const std::string& logbook_user,
                   const std::string& logbook_password,
                   const std::string& logbook_db,

                   const std::string& regdb_host,
                   const std::string& regdb_user,
                   const std::string& regdb_password,
                   const std::string& regdb_db,

                   const std::string& ifacedb_host,
                   const std::string& ifacedb_user,
                   const std::string& ifacedb_password,
                   const std::string& ifacedb_db ) throw (WrongParams,
                                                          DatabaseError)
{
    // Open and configure connections, and create the API object.
    //
    return new ConnectionImpl (
        connect2server ( logbook_host, logbook_user, logbook_password, logbook_db ),
        connect2server (   regdb_host,   regdb_user,   regdb_password,   regdb_db ),
        connect2server ( ifacedb_host, ifacedb_user, ifacedb_password, ifacedb_db )) ;
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
