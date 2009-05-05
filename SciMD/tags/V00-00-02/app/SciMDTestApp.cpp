//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class SciMDTestApp...
//
// Author List:
//      Igor Gaponenko
//
//------------------------------------------------------------------------

#include "Lusi/Lusi.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <list>
#include <string>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <exception>

#include <stdio.h>

//----------------------
// Base Class Headers --
//----------------------

#include "AppUtils/AppBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "AppUtils/AppCmdArg.h"
#include "AppUtils/AppCmdOpt.h"
#include "AppUtils/AppCmdOptIncr.h"
#include "AppUtils/AppCmdArgList.h"

#include "MsgLogger/MsgLogger.h"

#include "SciMD/Connection.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using std::cout;
using std::endl;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace SciMD {

/**
  * Exception class for signalling errors during parsing the command
  * arguments.
  */
class ParseException : public std::exception {

public:

    /**
      * Constructor
      */
    explicit ParseException (const std::string& message) :
        m_message ("SciMD::ParseException: " + message)
    {}

    /**
      * Destructor
      */
    virtual ~ParseException () throw ()
    {}

    /**
      * Report the cause.
      *
      * NOTE: The method is required by the base interface's contruct.
      */
    virtual const char* what () const throw ()
    {
        return m_message.c_str() ;
    }

private:

    // Data members
    //
    std::string m_message ;     // store the message to be reported
};

/**
  * Translate a string into a number.
  *
  * The method will throw an exception if the input string can't be
  * translated.
  *
  * @return a value
  */
int
str2int (const std::string& str) throw (SciMD::ParseException)
{
    if (str.empty())
        throw SciMD::ParseException ("empty string passed as an argument") ;

    int result = 0 ;
    if (1 != sscanf (str.c_str(), "%ud", &result))
        throw SciMD::ParseException ("the argument can't be translated") ;

    if (result < 0)
        throw SciMD::ParseException ("negative value of the argument isn't allowed") ;

    return result ;
}

/**
  * Application class.
  */
class SciMDTestApp : public AppUtils::AppBase {

public:

    /**
      * Constructor
      */
    explicit SciMDTestApp ( const std::string& appName ) ;

    /**
      * Destructor
      */
    ~SciMDTestApp () ;

protected:

    /**
      * Main method which runs the whole application.
      */
    virtual int runApp () ;

private:

    // Implement the commands

    int cmd_help () ;

    int cmd_add_run () throw (std::exception) ;

    int cmd_set_run_param () throw (std::exception) ;

    int cmd_param_info () throw (std::exception) ;

private:

    // Command line options and arguments
    //
    AppUtils::AppCmdArg<std::string >     m_command ;
    AppUtils::AppCmdArgList<std::string > m_args ;

    AppUtils::AppCmdOpt<std::string >  m_source ;
    AppUtils::AppCmdOpt<std::string >  m_odbc_conn ;
    AppUtils::AppCmdOptIncr            m_update ;

    // Database connection
    //
    SciMD::Connection* m_connection ;
};


// --------------------
// BEGIN IMPLEMENTATION 
// --------------------

SciMDTestApp::SciMDTestApp (const std::string& appName) :
    AppUtils::AppBase (appName),
    m_command ("command",
               "command name"),
    m_args ("arguments",
            "command specific arguments; use command 'help' for detail",
            std::list<std::string >()),
    m_source ('s',
              "source",
              "string",
              "a source of the modification",
              "TEST"),
    m_odbc_conn ('c',
                 "odbc-conn",
                 "string",
                 "ODBC connection string",
                 "DSN=SCIMD"),
    m_update ('u',
              "update",
              "update is allowed",
              0),
    m_connection (0)
{
    addArgument (m_command) ;
    addArgument (m_args) ;
    addOption   (m_source) ;
    addOption   (m_odbc_conn) ;
    addOption   (m_update) ;
}

SciMDTestApp::~SciMDTestApp ()
{
    delete m_connection ;
    m_connection = 0 ;
}

int
SciMDTestApp::runApp ()
{
    try {

        // Parse the arguments of the command
        //
        const std::string command = m_command.value();

        // Quick processing for the information command(s) for which
        // we aren't making any connections.
        //
        if (command == "help" ) return cmd_help();

        // Connect to the database
        //
        m_connection = SciMD::Connection::open (m_odbc_conn.value()) ;
        if (!m_connection) {
            MsgLogRoot( error, "failed to connect to: " << m_odbc_conn.value()) ;
            return 2 ;
        }

        // Proceed to commands which require the database
        //
        if      (command == "add_run")       return cmd_add_run ();
        else if (command == "set_run_param") return cmd_set_run_param ();
        else if (command == "param_info")    return cmd_param_info ();
        else {
            MsgLogRoot( error, "unknown command") ;
            return 2 ;
        }

    } catch ( const std::exception& e ) {

        MsgLogRoot( error, "exception caught: " << e.what() ) ;
        return 2 ;

    } catch ( ... ) {

        MsgLogRoot( error, "unknown exception caught" ) ;
        return 2 ;

    }
    return 0 ;
}

int
SciMDTestApp::cmd_help ()
{
    if (m_args.empty()) {
        cout << "SUPPORTED COMMANDS & ARGUMENTS:\n"
             << "\n"
             << "  help [command]\n"
             << "\n"
             << "  add_run       <experiment> <run> <begn_time> <end_time>\n"
             << "  set_run_param <experiment> <run> <param> <value>\n"
             << "  param_info    <experiment> <param>" << endl;
        return 0 ;
    }
    if (m_args.size() != 1) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }

    const std::string command = *(m_args.begin());
    if (command == "help") {

        cout << "SYNTAX:\n"
             << "\n"
             << "  help [command]\n"
             << "\n"
             << "DESCRIPTION:\n"
             << "\n"
             << "  Print an information on supported commands. If no command\n"
             << "  name is given then a general syntax of each command will\n"
             << "  be reported. Otherwise a full description of the command\n"
             << "  in question will be printed." << endl ;

    } else if (command == "add_run") {

        cout << "SYNTAX:\n"
             << "\n"
             << "  add_run <experiment> <run> <type> <begn_time> <end_time>\n"
             << "\n"
             << "DESCRIPTION:\n"
             << "\n"
             << "  Add a new run to the database for the experiment. The experiment\n"
             << "  is given by its name.\n"
             << "\n"
             << "  The run number must be a positive number.\n"
             << "\n"
             << "  For the run type use one of the predefined names (not available\n"
             << "  for this application).\n"
             << "\n"
             << "  The begin and end times specify a duration interval of the run.\n"
             << "  The duration interval must be contained within an interval of\n"
             << "  the experiment. Value of the timestamps are expected to have\n"
             << "  the following syntax:\n"
             << "\n"
             << "    YYYY-MM-DD HH:MM::SS\n"
             << "\n"
             << "  For example:\n"
             << "     2009-APR-27 16:23:05" << endl ;

    } else if (command == "set_run_param") {

        cout << "SYNTAX:\n"
             << "\n"
             << "  set_run_param <experiment> <run> <param> <value>\n"
             << "\n"
             << "DESCRIPTION:\n"
             << "\n"
             << "  Set/update a value of the requested parameter. The parameter\n"
             << "  has to be already configured in the database for the experiment.\n"
             << "\n"
             << "  The value of the parameter must be of the same type used\n"
             << "  during the parameter's configuration.\n"
             << "\n"
             << "  The 'source' of the parameter's value can be specified using\n"
             << "  the '-s' option. The default value of the 'source' is: "
             << m_source.defValue() << endl ;

    } else if (command == "param_info") {

        cout << "SYNTAX:\n"
             << "\n"
             << "  param_info <experiment> <param>\n"
             << "\n"
             << "DESCRIPTION:\n"
             << "\n"
             << "  Check if the specified parameter exists, and if so print its\n"
             << "  description.\n"
             << "\n"
             << "  Note, that the command won't display values of the parameter\n"
             << "  for runs." << endl ;

    } else {
        MsgLogRoot (error, "unknown command name requested") ;
        return 2 ;
    }
    return 0 ;
}

int
SciMDTestApp::cmd_add_run () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() != 5) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  experiment = *(itr++) ;
    const unsigned int run        = SciMD::str2int (*(itr++)) ;
    const std::string  type       = *(itr++) ;
    const std::string  begin_time = *(itr++) ;
    const std::string  end_time   = *(itr++) ;

    m_connection->beginTransaction () ;
    m_connection->createRun (
        experiment,
        run,
        type,
        begin_time,
        end_time) ;
    m_connection->commitTransaction () ;

    return 0 ;
}

int
SciMDTestApp::cmd_set_run_param () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() != 4) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  experiment = *(itr++) ;
    const int          run        = SciMD::str2int (*(itr++)) ;
    const std::string  param      = *(itr++) ;
    std::string        value      = *(itr++) ;

    m_connection->beginTransaction () ;
    m_connection->setRunParam (
        experiment,
        run,
        param,
        value,
        "SciMDTestApp",
        m_update.value () > 0) ;
    m_connection->commitTransaction () ;

    return 0 ;
}

int
SciMDTestApp::cmd_param_info () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() != 2) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  experiment = *(itr++) ;
    const std::string  param      = *(itr++) ;

    SciMD::ParamInfo p ;

    m_connection->beginTransaction () ;
    const bool exists = m_connection->getParamInfo (
        p,
        experiment,
        param) ;
    m_connection->commitTransaction () ;

    if (exists)
        cout << "       name: " << p.name << "\n"
             << " experiment: " << p.experiment << "\n"
             << "       type: " << p.type << "\n"
             << "description: " << p.descr << endl ;
    else
        cout << "Sorry, no such parameter!" << endl ;

    return 0 ;
}

} // namespace SciMD


// this defines main()
APPUTILS_MAIN(SciMD::SciMDTestApp)
