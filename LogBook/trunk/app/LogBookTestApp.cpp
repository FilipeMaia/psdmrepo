//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LogBookTestApp...
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

#include "LogBook/Connection.h"

#include "LusiTime/Time.h"
#include "LusiTime/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using std::cout;
using std::endl;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace LogBook {

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
        m_message ("LogBook::ParseException: " + message)
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
str2int (const std::string& str) throw (LogBook::ParseException)
{
    if (str.empty())
        throw LogBook::ParseException ("empty string passed as an argument") ;

    int result = 0 ;
    if (1 != sscanf (str.c_str(), "%ud", &result))
        throw LogBook::ParseException ("the argument can't be translated") ;

    if (result < 0)
        throw LogBook::ParseException ("negative value of the argument isn't allowed") ;

    return result ;
}

/**
  * Application class.
  */
class LogBookTestApp : public AppUtils::AppBase {

public:

    /**
      * Constructor
      */
    explicit LogBookTestApp ( const std::string& appName ) ;

    /**
      * Destructor
      */
    ~LogBookTestApp () ;

protected:

    /**
      * Main method which runs the whole application.
      */
    virtual int runApp () ;

private:

    // Implement the commands

    int cmd_help () ;
    int cmd_allocate_run () throw (std::exception) ;
    int cmd_add_run () throw (std::exception) ;
    int cmd_begin_run () throw (std::exception) ;
    int cmd_end_run () throw (std::exception) ;
    int cmd_set_run_param () throw (std::exception) ;
    int cmd_param_info () throw (std::exception) ;

private:

    // Command line options and arguments
    //
    AppUtils::AppCmdArg<std::string >     m_command ;
    AppUtils::AppCmdArgList<std::string > m_args ;

    AppUtils::AppCmdOpt<std::string >  m_source ;
    AppUtils::AppCmdOpt<std::string >  m_host ;
    AppUtils::AppCmdOpt<std::string >  m_user ;
    AppUtils::AppCmdOpt<std::string >  m_password ;
    AppUtils::AppCmdOptIncr            m_update ;

    // Database connection
    //
    LogBook::Connection* m_connection ;
};


// --------------------
// BEGIN IMPLEMENTATION 
// --------------------

LogBookTestApp::LogBookTestApp (const std::string& appName) :
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
    m_host ('H',
            "host",
            "string",
            "MySQL host to connect to",
            "localhost"),
    m_user ('U',
            "user",
            "string",
            "MySQL user account",
            ""),
    m_password ('P',
            "password",
            "string",
            "MySQL account password",
            ""),
    m_update ('u',
              "update",
              "update is allowed",
              0),
    m_connection (0)
{
    addArgument (m_command) ;
    addArgument (m_args) ;
    addOption   (m_source) ;
    addOption   (m_host) ;
    addOption   (m_user) ;
    addOption   (m_password) ;
    addOption   (m_update) ;
}

LogBookTestApp::~LogBookTestApp ()
{
    delete m_connection ;
    m_connection = 0 ;
}

int
LogBookTestApp::runApp ()
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
        m_connection = LogBook::Connection::open (
            m_host.value().c_str(),
            m_user.value().c_str(),
            m_password.value().c_str()) ;
        if (!m_connection) {
            MsgLogRoot( error, "failed to connect to the server" ) ;
            return 2 ;
        }

        // Proceed to commands which require the database
        //
        if      (command == "allocate_run")  return cmd_allocate_run ();
        else if (command == "add_run")       return cmd_add_run ();
        else if (command == "begin_run")     return cmd_begin_run ();
        else if (command == "end_run")       return cmd_end_run ();
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
LogBookTestApp::cmd_help ()
{
    if (m_args.empty()) {
        cout << "Usage: -h | help | <command>" << endl;
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
             << "SUPPORTED COMMANDS & ARGUMENTS:\n"
             << "\n"
             << "  allocate_run  <instrument> <experiment>\n"
             << "\n"
             << "  add_run       <instrument> <experiment> <run> {DATA|CALIB} <begn_time> <end_time>\n"
             << "  begin_run     <instrument> <experiment> <run> {DATA|CALIB} <begn_time>\n"
             << "  end_run       <instrument> <experiment> <run>                          <end_time>\n"
             << "\n"
             << "  set_run_param <instrument> <experiment> <run> <param> <value> {INT|DOUBLE|TEXT}\n"
             << "  param_info    <instrument> <experiment> <param>\n"
             << "\n"
             << "PARAMETERS:\n"
             << "\n"
             << "  - the instrument and experiment ate given by their names.\n"
             << "  - the run number must be a positive number.\n"
             << "  - values of the timestamps are expected to have the following syntax:\n"
             << "\n"
             << "      YYYY-MM-DD HH:MM::SS\n"
             << "\n"
             << "    For example:\n"
             << "\n"
             << "      2009-APR-27 16:23:05\n"
             << "\n"
             << "  - the run type can be one of the following:\n"
             << "\n"
             << "    'CALIB'\n"
             << "    'DATA'\n"
             << "\n"
             << "  - the run parameter type can be one of the following:\n"
             << "\n"
             << "      'INT'\n"
             << "      'DOUBLE'\n"
             << "      'TEXT'\n"
             << "\n"
             << "    Note, that values of run parameters must be of the same type used\n"
             << "    during the parameter's configuration.\n"
             << "\n"
             << "  - the 'source' of the parameter's value can be specified using\n"
             << "    the '-s' option. The default value of the 'source' is: " << m_source.defValue() << "\n"
             << endl ;

    } else if (command == "add_run") {

        cout << "SYNTAX:\n"
             << "\n"
             << "  add_run <instrument> <experiment> <run> <type> <begn_time> <end_time>\n"
             << "\n"
             << "DESCRIPTION:\n"
             << "\n"
             << "  Add a new run to the database for the experiment. Close a previous run\n"
             << "  if the one is still open."
             << endl ;

    } else if (command == "begin_run") {

        cout << "SYNTAX:\n"
             << "\n"
             << "  begin_run <instrument> <experiment> <run> <type> <begn_time>\n"
             << "\n"
             << "DESCRIPTION:\n"
             << "\n"
             << "  Begin the new run. The run will remain in the open-ended state and it shall\n"
             << "  be closed either explicitly or implicitly by starting another run."
             << endl ;

    } else if (command == "set_run_param") {

        cout << "SYNTAX:\n"
             << "\n"
             << "  set_run_param <instrument> <experiment> <run> <param> <value> <type>\n"
             << "\n"
             << "DESCRIPTION:\n"
             << "\n"
             << "  Set/update a value of the requested parameter. The parameter\n"
             << "  has to be already configured in the database for the experiment.\n"
             << "  Use the -u option to allow updates."
             << endl ;

    } else if (command == "param_info") {

        cout << "SYNTAX:\n"
             << "\n"
             << "  param_info <instrument> <experiment> <param>\n"
             << "\n"
             << "DESCRIPTION:\n"
             << "\n"
             << "  Check if the specified parameter exists, and if so print its\n"
             << "  description.\n"
             << "\n"
             << "  Note, that the command won't display values of the parameter\n"
             << "  for runs."
             << endl ;

    } else {
        MsgLogRoot (error, "unknown command name requested") ;
        return 2 ;
    }
    return 0 ;
}

int
LogBookTestApp::cmd_allocate_run () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() != 2) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  instrument = *(itr++) ;
    const std::string  experiment = *(itr++) ;

    try {
        m_connection->beginTransaction () ;
        int num = m_connection->allocateRunNumber (
            instrument,
            experiment) ;
        cout << "ALLOCATED RUN NUMBER : " << num << endl ;
        m_connection->commitTransaction () ;

    } catch (const LusiTime::Exception& e) {
         MsgLogRoot (error, e.what()) ;
    }
    return 0 ;
}

int
LogBookTestApp::cmd_add_run () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() != 6) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  instrument = *(itr++) ;
    const std::string  experiment = *(itr++) ;
    const unsigned int run        = LogBook::str2int (*(itr++)) ;
    const std::string  type       = *(itr++) ;
    const std::string  begin_time = *(itr++) ;
    const std::string  end_time   = *(itr++) ;

    try {
        m_connection->beginTransaction () ;
        m_connection->createRun (
            instrument,
            experiment,
            run,
            type,
            LusiTime::Time::parse (begin_time),
            LusiTime::Time::parse (  end_time)) ;
        m_connection->commitTransaction () ;

    } catch (const LusiTime::Exception& e) {
         MsgLogRoot (error, e.what()) ;
    }
    return 0 ;
}

int
LogBookTestApp::cmd_begin_run () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() != 5) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  instrument = *(itr++) ;
    const std::string  experiment = *(itr++) ;
    const unsigned int run        = LogBook::str2int (*(itr++)) ;
    const std::string  type       = *(itr++) ;
    const std::string  begin_time = *(itr++) ;

    try {
        m_connection->beginTransaction () ;
        m_connection->beginRun (
            instrument,
            experiment,
            run,
            type,
            LusiTime::Time::parse (begin_time)) ;
        m_connection->commitTransaction () ;

    } catch (const LusiTime::Exception& e) {
         MsgLogRoot (error, e.what()) ;
    }

    return 0 ;
}

int
LogBookTestApp::cmd_end_run () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() != 4) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  instrument = *(itr++) ;
    const std::string  experiment = *(itr++) ;
    const unsigned int run        = LogBook::str2int (*(itr++)) ;
    const std::string  end_time   = *(itr++) ;

    try {
        m_connection->beginTransaction () ;
        m_connection->endRun (
            instrument,
            experiment,
            run,
            LusiTime::Time::parse (end_time)) ;
        m_connection->commitTransaction () ;

    } catch (const LusiTime::Exception& e) {
         MsgLogRoot (error, e.what()) ;
    }

    return 0 ;
}

int
LogBookTestApp::cmd_set_run_param () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() != 6) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  instrument = *(itr++) ;
    const std::string  experiment = *(itr++) ;
    const int          run        = LogBook::str2int (*(itr++)) ;
    const std::string  param      = *(itr++) ;
    std::string        value      = *(itr++) ;
    const  std::string type       = *(itr++) ;

    if (type == "INT") {

        int value_int ;
        if (1 != sscanf(value.c_str(), "%d", &value_int)) {
            MsgLogRoot (error, "parameter value is not of the claimed type") ;
            return 2 ;
        }
        m_connection->beginTransaction () ;
        m_connection->setRunParam (
            instrument,
            experiment,
            run,
            param,
            value_int,
            "LogBookTestApp",
            m_update.value () > 0) ;
        m_connection->commitTransaction () ;

    } else if (type == "DOUBLE") {

        double value_double ;
        if (1 != sscanf(value.c_str(), "%lf", &value_double)) {
            MsgLogRoot (error, "parameter value is not of the claimed type") ;
            return 2 ;
        }
        m_connection->beginTransaction () ;
        m_connection->setRunParam (
            instrument,
            experiment,
            run,
            param,
            value_double,
            "LogBookTestApp",
            m_update.value () > 0) ;
        m_connection->commitTransaction () ;

    } else if (type == "TEXT") {

        m_connection->beginTransaction () ;
        m_connection->setRunParam (
            instrument,
            experiment,
            run,
            param,
            value,
            "LogBookTestApp",
            m_update.value () > 0) ;
        m_connection->commitTransaction () ;

    } else {
        MsgLogRoot (error, "unsupported type of the value") ;
        return 2 ;
    }
    return 0 ;
}

int
LogBookTestApp::cmd_param_info () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() != 3) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  instrument = *(itr++) ;
    const std::string  experiment = *(itr++) ;
    const std::string  param      = *(itr++) ;

    LogBook::ParamInfo p ;

    m_connection->beginTransaction () ;
    const bool exists = m_connection->getParamInfo (
        p,
        instrument,
        experiment,
        param) ;
    m_connection->commitTransaction () ;

    if (exists)
        cout << "       name: " << p.name << "\n"
             << " instrument: " << p.instrument << "\n"
             << " experiment: " << p.experiment << "\n"
             << "       type: " << p.type << "\n"
             << "description: " << p.descr << endl ;
    else
        cout << "Sorry, no such parameter!" << endl ;

    return 0 ;
}

} // namespace LogBook


// this defines main()
APPUTILS_MAIN(LogBook::LogBookTestApp)
