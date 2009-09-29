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
#include <iomanip>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <exception>

#include <stdio.h>
#include <strings.h>

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

    int cmd_experiments       () throw (std::exception) ;

    int cmd_allocate_run      () throw (std::exception) ;
    int cmd_add_run           () throw (std::exception) ;
    int cmd_begin_run         () throw (std::exception) ;
    int cmd_end_run           () throw (std::exception) ;

    int cmd_create_run_param  () throw (std::exception) ;
    int cmd_param_info        () throw (std::exception) ;
    int cmd_run_parameters    () throw (std::exception) ;

    int cmd_set_run_param     () throw (std::exception) ;
    int cmd_display_run_param () throw (std::exception) ;

    int cmd_save_files        () throw (std::exception) ;
    int cmd_save_files_m      () throw (std::exception) ;

private:

    // Command line options and arguments
    //
    AppUtils::AppCmdArg<std::string >     m_command ;
    AppUtils::AppCmdArgList<std::string > m_args ;

    AppUtils::AppCmdOpt<std::string >  m_source ;
    AppUtils::AppCmdOpt<std::string >  m_config_file ;
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
    m_config_file ('c',
                   "config-file",
                   "string",
                   "Configuration file with MySQL connection parameters",
                   ""),
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
    addOption   (m_config_file) ;
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
        if( !m_config_file.value().empty())
            m_connection = LogBook::Connection::open ( m_config_file.value()) ;
        else
            m_connection = LogBook::Connection::open (

                m_host.value().c_str(),
                m_user.value().c_str(),
                m_password.value().c_str(),
                "LogBook",

                m_host.value().c_str(),
                m_user.value().c_str(),
                m_password.value().c_str(),
                "RegDB",

                m_host.value().c_str(),
                m_user.value().c_str(),
                m_password.value().c_str(),
                "interface_db"
            ) ;
        if (!m_connection) {
            MsgLogRoot( error, "failed to connect to the server" ) ;
            return 2 ;
        }

        // Proceed to commands which require the database
        //
        if      (command == "experiments")       return cmd_experiments();
        else if (command == "allocate_run")      return cmd_allocate_run ();
        else if (command == "add_run")           return cmd_add_run ();
        else if (command == "begin_run")         return cmd_begin_run ();
        else if (command == "end_run")           return cmd_end_run ();
        else if (command == "create_run_param")  return cmd_create_run_param ();
        else if (command == "param_info")        return cmd_param_info ();
        else if (command == "run_parameters")    return cmd_run_parameters ();
        else if (command == "set_run_param")     return cmd_set_run_param ();
        else if (command == "display_run_param") return cmd_display_run_param ();
        else if (command == "save_files")        return cmd_save_files ();
        else if (command == "save_files_m")      return cmd_save_files_m ();
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
    cout << "SYNTAX:\n"
         << "\n"
         << "  experiments   [<instrument>]\n"
         << "\n"
         << "  allocate_run  <instrument> <experiment>\n"
         << "\n"
         << "  add_run       <instrument> <experiment> <run> {DATA|CALIB} <begn_time> <end_time>\n"
         << "  begin_run     <instrument> <experiment> <run> {DATA|CALIB} <begn_time>\n"
         << "  end_run       <instrument> <experiment> <run>                          <end_time>\n"
         << "\n"
         << "  create_run_param  <instrument> <experiment> <param> {INT|DOUBLE|TEXT} <description>\n"
         << "  param_info        <instrument> <experiment> <param>\n"
         << "  run_parameters    <instrument> <experiment>\n"
         << "\n"
         << "  set_run_param     <instrument> <experiment> <run> <param> <value> {INT|DOUBLE|TEXT}\n"
         << "  display_run_param <instrument> <experiment> <run> <param>\n"
         << "\n"
         << "  save_files       <instrument> <experiment> <run> {DATA|CALIB}\n"
         << "  save_files_m     <instrument> <experiment> <run> {DATA|CALIB} [ <file1> {XTC|EPICS} ] [ <file>2 {XTC|EPICS} ] ...\n"
         << "\n"
         << "NOTES ON PARAMETERS:\n"
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
         << "    when creating (definig) the parameter.\n"
         << "\n"
         << "  - the 'source' of the parameter's value can be specified using\n"
         << "    the '-s' option. The default value of the 'source' is: " << m_source.defValue() << "\n"
         << "\n"
         << "COMMANDS:\n"
         << "\n"
         << "  add_run\n"
         << "\n"
         << "    - add a new run to the database for the experiment. Close a previous run\n"
         << "      if the one is still open.\n"
         << "\n"
         << "  begin_run\n"
         << "\n"
         << "    - begin the new run. The run will remain in the open-ended state and it shall\n"
         << "      be closed either explicitly or implicitly by starting another run.\n"
         << "\n"
         << "  create_run_param\n"
         << "\n"
         << "    - create (define) a new run parameter for an experiment. Values of the parameter\n"
         << "      are set for each run independently by calling the 'set_run_param' command.\n"
         << "\n"
         << "  param_info\n"
         << "\n"
         << "    - check if the specified parameter exists, and if so print its\n"
         << "      description. Note, that the command won't display values of\n"
         << "      the parameters for runs. Use the 'display_run_param' command to\n"
         << "      display its value for a run if set.\n"
         << "\n"
         << "  run_parameters\n"
         << "\n"
         << "    - locate and display definitions of all prun parameters of an experiment.\n"
         << "      Note, that the command won't display values of\n"
         << "      the parameters for runs. Use the 'display_run_param' command to\n"
         << "      display its value for a run if set.\n"
         << "\n"
         << "  set_run_param\n"
         << "\n"
         << "    - set/update a value of the requested parameter. The parameter\n"
         << "      has to be already configured in the database for the experiment.\n"
         << "      Use the -u option to allow updates.\n"
         << "\n"
         << "  display_run_param\n"
         << "\n"
         << "    - display a value (if any) of a run parameter set for a run.\n"
         << "\n"
         << "  save_files\n"
         << "\n"
         << "    - tell OFFLINE to further process (translate from the XTC into HDF5\n"
         << "      representation, archive, etc.) data files of a run.\n"
         << "\n"
         << "  save_files_m\n"
         << "\n"
         << "    - tell OFFLINE to further process (translate from the XTC into HDF5\n"
         << "      representation, archive, etc.) data files of a run. Unlike the previous\n"
         << "      command this one will also register the specified files in the data set\n"
         << "      associated with the run.\n"
         << endl ;

    return 0 ;
}

int
LogBookTestApp::cmd_experiments () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.size() > 1 ) {
        MsgLogRoot (error, "wrong number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    std::string instrument = "" ;
    if (itr != m_args.end()) instrument = *(itr++) ;

    m_connection->beginTransaction () ;
    std::vector<LogBook::ExperDescr > experiments ;
    m_connection->getExperiments (
        experiments,
        instrument) ;
    for (size_t i = 0 ; i < experiments.size(); i++)
        cout << "\n"
             << experiments[i];
    m_connection->commitTransaction () ;

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
LogBookTestApp::cmd_create_run_param () throw (std::exception)
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
    const std::string  param      = *(itr++) ;
    const std::string  type       = *(itr++) ;
    const std::string  descr      = *(itr++) ;

    // Check if such parameter already exists. Complain if it does.
    //
    int status = 0 ;

    m_connection->beginTransaction () ;

    LogBook::ParamInfo p ;
    if (m_connection->getParamInfo (
        p,
        instrument,
        experiment,
        param)) {
            cout << "Sorry, the parameter already exists in the database." << endl ;
            status = 1 ;
    } else {
        m_connection->createRunParam (
            instrument,
            experiment,
            param,
            type,
            descr) ;
    }
    m_connection->commitTransaction () ;

    return status ;
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

    m_connection->beginTransaction () ;

    LogBook::ParamInfo p ;
    if (m_connection->getParamInfo (
        p,
        instrument,
        experiment,
        param))
        cout << p << endl ;
    else
        cout << "Sorry, no such parameter." << endl ;

    m_connection->commitTransaction () ;

    return 0 ;
}

int
LogBookTestApp::cmd_run_parameters () throw (std::exception)
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

    // Get the parameters
    //
    std::vector<LogBook::ParamInfo > params ;

    m_connection->beginTransaction () ;
    m_connection->getParamsInfo (
        params,
        instrument,
        experiment) ;
    m_connection->commitTransaction () ;

    // Calculate "pretty print" parameters and print results
    //
    size_t name_len  = strlen ("Name") ;
    size_t type_len  = strlen ("Type") ;
    size_t descr_len = strlen ("Description") ;
    for (size_t i = 0 ; i < params.size (); i++) {
        if (params[i].name.size  () > name_len)  name_len = params[i].name.size () ;
        if (params[i].type.size  () > type_len)  type_len = params[i].type.size () ;
        if (params[i].descr.size () > descr_len) descr_len = params[i].descr.size () ;
    }
    if (descr_len > 80) descr_len = 80 ;
    cout << "\n" << std::left
         << "  " << std::setw (name_len) << "Name" << " | " << std::setw (type_len) << "Type" << " | Description\n"
         << " " << std::string (name_len+2, '-') << "+" << std::string (type_len+2, '-') << "+" << std::string (descr_len+2, '-') << "\n";
    for (size_t i = 0 ; i < params.size (); i++)
        cout << "  " << std::setw (name_len) << params[i].name << " | " << std::setw (type_len) << params[i].type << " | " << std::setw (descr_len) << params[i].descr << "\n" ;
    cout << endl ;

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

    if (!strcasecmp ("INT", type.c_str())) {

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

    } else if (!strcasecmp ("DOUBLE", type.c_str())) {

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

    } else if (!strcasecmp ("TEXT", type.c_str())) {

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
LogBookTestApp::cmd_display_run_param () throw (std::exception)
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
    const int          run        = LogBook::str2int (*(itr++)) ;
    const std::string  param      = *(itr++) ;


    // Find the parameter to be sure it exists and also to learn its type.
    //
    m_connection->beginTransaction () ;

    LogBook::ParamInfo p ;
    if (!m_connection->getParamInfo ( p,
                                      instrument,
                                      experiment,
                                      param)) {
        cout << "Sorry, no such parameter in the database." << endl ;
        m_connection->commitTransaction () ;
        return 1 ;
    }

    // Get a value (if set) of the parameter for the run.
    //
    std::string    source ;
    LusiTime::Time updated ;

    if (p.type == "INT") {

        int value ;
        m_connection->getRunParam (
            instrument,
            experiment,
            run,
            param,
            value,
            source,
            updated) ;

        cout << "VALUE:   " << value << "\n";

    } else if (p.type == "DOUBLE") {

        double value ;
        m_connection->getRunParam (
            instrument,
            experiment,
            run,
            param,
            value,
            source,
            updated) ;

        cout << "VALUE:   " << value << "\n";

    } else if (p.type == "TEXT") {

        std::string value ;
        m_connection->getRunParam (
            instrument,
            experiment,
            run,
            param,
            value,
            source,
            updated) ;

        cout << "VALUE:   " << value << "\n";

    } else {
        cout << "Sorry, no support for the parameter type: " << p.type << endl ;
        m_connection->commitTransaction () ;
        return 1 ;
    }
    m_connection->commitTransaction () ;

    cout << "SOURCE:  " << source << "\n";
    cout << "UPDATED: " << updated << "\n";

    return 0 ;
}

int
LogBookTestApp::cmd_save_files () throw (std::exception)
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
    const std::string  run_type   = *(itr++) ;

    m_connection->beginTransaction () ;
    m_connection->saveFiles (
        instrument,
        experiment,
        run,
        run_type) ;
    m_connection->commitTransaction () ;

    return 0 ;
}

int
LogBookTestApp::cmd_save_files_m () throw (std::exception)
{
    // Parse and verify the arguments
    //
    if (m_args.empty() || m_args.size() < 4 || m_args.size() % 2) {
        MsgLogRoot (error, "insufficient number of arguments to the command") ;
        return 2 ;
    }
    AppUtils::AppCmdArgList<std::string >::const_iterator itr = m_args.begin() ;
    const std::string  instrument = *(itr++) ;
    const std::string  experiment = *(itr++) ;
    const unsigned int run        = LogBook::str2int (*(itr++)) ;
    const std::string  run_type   = *(itr++) ;
    const size_t       num_files  = (m_args.size() - 4) / 2 ;
    std::vector<std::string > files      (num_files) ;
    std::vector<std::string > file_types (num_files) ;
    for (size_t i = 0; i < num_files; i++) {
        const std::string file = *(itr++) ;
        const std::string type = *(itr++) ;
        files      [i] = file ;
        file_types [i] = type ;
    }

    m_connection->beginTransaction () ;
    m_connection->saveFiles (
        instrument,
        experiment,
        run,
        run_type,
        files,
        file_types) ;
    m_connection->commitTransaction () ;

    return 0 ;
}

} // namespace LogBook


// this defines main()
APPUTILS_MAIN(LogBook::LogBookTestApp)
