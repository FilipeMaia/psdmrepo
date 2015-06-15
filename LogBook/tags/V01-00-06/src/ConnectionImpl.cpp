//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: $
//
// Description:
//	Class ConnectionImpl...
//
// Author List:
//      Igor Gaponenko
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "LogBook/ConnectionImpl.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <strings.h>
#include <stdlib.h>
#include <ctype.h>

#include <memory>
#include <iostream>
using namespace std ;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "LogBook/QueryProcessor.h"

#include "LusiTime/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace LogBook {

inline
void
string2upper (std::string& out)
{
    for (size_t i = 0; i < out.length(); i++)
        out[i] = toupper (out[i]) ;
}

inline
bool
isValidRunType (const std::string& type)
{
    static const char* const validTypes[] = {"DATA", "CALIB", "TEST", "EPICS"} ;
    static const size_t numTypes = 2 ;
    for (size_t i = 0 ; i < numTypes ; ++i)
        if (0 == strcasecmp(validTypes[i], type.c_str()))
            return true ;
    return false ;
}

inline
bool
isValidFileType (const std::string& type)
{
    static const char* const validTypes[] = {"XTC", "EPICS"} ;
    static const size_t numTypes = 2 ;
    for (size_t i = 0 ; i < numTypes ; ++i)
        if (0 == strcasecmp(validTypes[i], type.c_str()))
            return true ;
    return false ;
}

inline
bool
isValidValueType (const std::string& type)
{
    static const char* const validTypes[] = {"INT", "DOUBLE", "TEXT"} ;
    static const size_t numTypes = 3 ;
    for (size_t i = 0 ; i < numTypes ; ++i)
        if (0 == strcasecmp(validTypes[i], type.c_str()))
            return true ;
    return false ;
}

inline
void
row2attr (AttrInfo& info,
          QueryProcessor& query,
          const std::string& instrument, const std::string& experiment, int run)
{
    info.instrument = instrument ;
    info.experiment = experiment ;
    info.run        = run ;

    query.get (info.attr_class, "class") ;
    query.get (info.attr_name,  "name") ;
    query.get (info.attr_type,  "type") ;
    query.get (info.attr_descr, "descr", true) ;
}

//-------------
// Operators --
//-------------

std::ostream&
operator<< (std::ostream& s, const ParamDescr& d)
{
    s << "LogBook::ParamDescr {\n"
      << "          id: " << d.id << "\n"
      << "        name: " << d.name << "\n"
      << "    exper_id: " << d.exper_id << "\n"
      << "        type: " << d.type << "\n"
      << "       descr: " << d.descr << "\n"
      << "}\n" ;
    return s ;
}

std::ostream&
operator<< (std::ostream& s, const RunDescr& d)
{
    s << "LogBook::RunDescr {\n"
      << "          id: " << d.id << "\n"
      << "         num: " << d.num << "\n"
      << "    exper_id: " << d.exper_id << "\n"
      << "        type: " << d.type << "\n"
      << "  begin_time: " << d.begin_time << "\n"
      << "    end_time: " << d.end_time << "\n"
      << "}\n" ;
    return s ;
}

//----------------
// Constructors --
//----------------

ConnectionImpl::ConnectionImpl (MYSQL* logbook_mysql,
                                MYSQL*   regdb_mysql,
                                MYSQL* ifacedb_mysql) :
    Connection () ,
    m_is_started    (false) ,
    m_logbook_mysql (logbook_mysql) ,
    m_regdb_mysql   (regdb_mysql) ,
    m_ifacedb_mysql (ifacedb_mysql)
{}

//--------------
// Destructor --
//--------------

ConnectionImpl::~ConnectionImpl () throw ()
{
    mysql_close( m_logbook_mysql ) ;
    mysql_close( m_regdb_mysql ) ;
    mysql_close( m_ifacedb_mysql ) ;

    m_logbook_mysql = 0 ;
    m_regdb_mysql   = 0 ;
    m_ifacedb_mysql = 0 ;
}

//-----------
// Methods --
//-----------

void
ConnectionImpl::beginTransaction () throw (DatabaseError)
{
    if (m_is_started) return ;
    this->simpleQuery (m_logbook_mysql, "BEGIN");
    this->simpleQuery (m_regdb_mysql,   "BEGIN");
    this->simpleQuery (m_ifacedb_mysql, "BEGIN");
    m_is_started = true ;
}

void
ConnectionImpl::commitTransaction () throw (DatabaseError)
{
    if (!m_is_started) return ;
    this->simpleQuery (m_logbook_mysql, "COMMIT");
    this->simpleQuery (m_regdb_mysql,   "COMMIT");
    this->simpleQuery (m_ifacedb_mysql, "COMMIT");
    m_is_started = false ;
}

void
ConnectionImpl::abortTransaction () throw (DatabaseError)
{
    if (!m_is_started) return ;
    this->simpleQuery (m_logbook_mysql, "ROLLBACK");
    this->simpleQuery (m_regdb_mysql,   "ROLLBACK");
    this->simpleQuery (m_ifacedb_mysql, "ROLLBACK");
    m_is_started = false ;
}

bool
ConnectionImpl::transactionIsStarted () const
{
    return m_is_started ;
}

bool
ConnectionImpl::getCurrentExperiment (ExperDescr&        descr,
                                      const std::string& instrument,
                                      unsigned int       station) throw (WrongParams,
                                                                         DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    if (instrument.empty())
        throw WrongParams ("intrument name can't be empty") ;

    // Get the identifier of the current experiment first (if any)
    //
    int exper_id = 0 ;
    if( !this->getCurrentExperimentId ( exper_id, instrument, station )) return false;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT i.name AS 'instr_name',i.descr AS 'instr_descr',e.* FROM "
        << "instrument i, "
        << "experiment e WHERE e.instr_id=i.id"
        << " AND e.id=" << exper_id ;

    QueryProcessor query (m_regdb_mysql) ;
    query.execute (sql.str()) ;

    // Extract results
    //
    if (!query.next_row())
        throw DatabaseError ("unable to get experiment description for the current experiment");

    query.get (descr.instr_id,    "instr_id") ;
    query.get (descr.instr_name,  "instr_name") ;
    query.get (descr.instr_descr, "instr_descr") ;

    query.get (descr.id,    "id") ;
    query.get (descr.name,  "name") ;
    query.get (descr.descr, "descr") ;

    query.get (descr.registration_time, "registration_time") ;
    query.get (descr.begin_time,        "begin_time") ;
    query.get (descr.end_time,          "end_time") ;

    query.get (descr.leader_account, "leader_account") ;
    query.get (descr.contact_info,   "contact_info") ;
    query.get (descr.posix_gid,      "posix_gid") ;

    return true;
}

bool
ConnectionImpl::getCurrentExperimentId (int&               id,
                                        const std::string& instrument,
                                        unsigned int       station) throw (WrongParams,
                                                                           DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    if (instrument.empty())
        throw WrongParams ("intrument name can't be empty") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT exper_id FROM expswitch"
        << " WHERE exper_id IN ("
        << "   SELECT experiment.id FROM experiment, instrument"
        << "   WHERE experiment.instr_id=instrument.id AND instrument.name='" << instrument
        << "')"
        << " AND station=" << station
        << " ORDER BY switch_time DESC LIMIT 1" ;

    QueryProcessor query (m_regdb_mysql) ;
    query.execute (sql.str()) ;

    // Extract results
    //
    if (!query.next_row()) return false ;

    query.get (id, "exper_id") ;

    return true ;
}


void
ConnectionImpl::getExperiments (std::vector<ExperDescr >& experiments,
                                const std::string&        instrument) throw (WrongParams,
                                                                             DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT i.name AS 'instr_name',i.descr AS 'instr_descr',e.* FROM "
        << "instrument i, "
        << "experiment e WHERE e.instr_id=i.id" ;
    if (instrument != "")
        sql << " AND i.name='" << instrument << "'";

    QueryProcessor query (m_regdb_mysql) ;
    query.execute (sql.str()) ;

    // Extract results
    //
    experiments.clear () ;

    while (query.next_row()) {

        ExperDescr descr ;

        query.get (descr.instr_id,    "instr_id") ;
        query.get (descr.instr_name,  "instr_name") ;
        query.get (descr.instr_descr, "instr_descr") ;

        query.get (descr.id,    "id") ;
        query.get (descr.name,  "name") ;
        query.get (descr.descr, "descr") ;

        query.get (descr.registration_time, "registration_time") ;
        query.get (descr.begin_time,        "begin_time") ;
        query.get (descr.end_time,          "end_time") ;

        query.get (descr.leader_account, "leader_account") ;
        query.get (descr.contact_info,   "contact_info") ;
        query.get (descr.posix_gid,      "posix_gid") ;

        experiments.push_back (descr) ;
    }
}

bool
ConnectionImpl::getOneExperiment (ExperDescr&        descr,
                                  const std::string& instrument,
                                  const std::string& experiment) throw (WrongParams,
                                                                        DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT i.name AS 'instr_name',i.descr AS 'instr_descr',e.* FROM "
        << "instrument i, "
        << "experiment e WHERE e.instr_id=i.id"
        << " AND i.name='" << instrument << "'"
        << " AND e.name='" << experiment << "'" ;

    QueryProcessor query (m_regdb_mysql) ;
    query.execute (sql.str()) ;

    // Extract results
    //
    if (!query.next_row()) return false ;

    query.get (descr.instr_id,    "instr_id") ;
    query.get (descr.instr_name,  "instr_name") ;
    query.get (descr.instr_descr, "instr_descr") ;

    query.get (descr.id,    "id") ;
    query.get (descr.name,  "name") ;
    query.get (descr.descr, "descr") ;

    query.get (descr.registration_time, "registration_time") ;
    query.get (descr.begin_time,        "begin_time") ;
    query.get (descr.end_time,          "end_time") ;

    query.get (descr.leader_account, "leader_account") ;
    query.get (descr.contact_info,   "contact_info") ;
    query.get (descr.posix_gid,      "posix_gid") ;

    return true ;
}

bool
ConnectionImpl::getParamInfo (ParamInfo&         info,
                              const std::string& instrument,
                              const std::string& experiment,
                              const std::string& parameter) throw (WrongParams,
                                                                   DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    ParamDescr param_descr ;
    if (!this->findRunParam (param_descr, exper_descr.id, parameter))
        return false ;

    info.name       = parameter ;
    info.instrument = instrument ;
    info.experiment = experiment ;
    info.type       = param_descr.type ;
    info.descr      = param_descr.descr ;

    return true ;
}


void
ConnectionImpl::getParamsInfo (std::vector<ParamInfo >& info,
                               const std::string&       instrument,
                               const std::string&       experiment) throw (WrongParams,
                                                                          DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT * FROM run_param WHERE exper_id=" << exper_descr.id ;

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    // Extract results
    //
    info.reserve (query.num_rows ()) ;

    while (query.next_row()) {

        ParamInfo param ;

        param.instrument = instrument ;
        param.experiment = experiment ;

        query.get (param.name,     "param") ;
        query.get (param.type,     "type") ;
        query.get (param.descr,    "descr", true) ;

        info.push_back (param) ;
    }
}

int
ConnectionImpl::allocateRunNumber (const std::string& instrument,
                                   const std::string& experiment) throw (WrongParams,
                                                                         DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    try {

        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, instrument, experiment))
            throw WrongParams ("unknown experiment") ;

        // The current timestamp will be recorded as a time when the run umber
        // was requested/allocated.
        //
        const LusiTime::Time now = LusiTime::Time::now () ;

        // Now proceed with the new run allocation
        //
        std::ostringstream sql;
        sql << "INSERT INTO run_" << exper_descr.id
            << " VALUES(NULL," << LusiTime::Time::to64 (now) << ")";

        this->simpleQuery (m_regdb_mysql, sql.str());

        // Get back its number
        //
        QueryProcessor query (m_regdb_mysql) ;
        query.execute ("SELECT LAST_INSERT_ID() AS 'num'") ;
        if (!query.next_row())
            throw DatabaseError ("inconsistent result from the database") ;

        int num = 0 ;
        query.get (num, "num") ;
        return num ;

    } catch (const LusiTime::Exception& e) {
        throw WrongParams (
            std::string ("failed to translate LusiTime::Time to string because of: ")
            + e.what()) ;
    }
}

void
ConnectionImpl::createRun (const std::string&    instrument,
                           const std::string&    experiment,
                           int                   run,
                           const std::string&    run_type,
                           const LusiTime::Time& beginTime,
                           const LusiTime::Time& endTime) throw (WrongParams,
                                                                 DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    if (!beginTime.isValid ())
        throw WrongParams ("the begin run timstamp isn't valid") ;

    std::string type = run_type ;
    string2upper (type) ;
    if (!LogBook::isValidRunType (type))
        throw WrongParams ("unknown run type: "+run_type) ;

    try {

        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, instrument, experiment))
            throw WrongParams ("unknown experiment") ;

        // Find out the previous run (if any) and make sure it gets closed
        // if it's still being open.
        //
        RunDescr run_descr ;
        if (this->findLastRun (run_descr, exper_descr.id)) {
            if (!run_descr.end_time.isValid()) {
                this->endRun (instrument, experiment, run_descr.num, beginTime) ;
            }
        }

        // Now proceed with the new run creation
        //
        std::ostringstream sql;
        sql << "INSERT INTO run VALUES(NULL,"
            << run << ","
            << exper_descr.id << ",'"
            << type << "',"
            << LusiTime::Time::to64 (beginTime) << ",";

        if (endTime.isValid ())
            sql << LusiTime::Time::to64 (endTime) << ")";
        else
            sql << "NULL)";

        this->simpleQuery (m_logbook_mysql, sql.str());

    } catch (const LusiTime::Exception& e) {
        throw WrongParams (
            std::string ("failed to translate LusiTime::Time to string because of: ")
            + e.what()) ;
    }
}

void
ConnectionImpl::beginRun (const std::string&    instrument,
                          const std::string&    experiment,
                          int                   run,
                          const std::string&    run_type,
                          const LusiTime::Time& beginTime) throw (WrongParams,
                                                                  DatabaseError)
{
    this->createRun (instrument,
                     experiment,
                     run,
                     run_type,
                     beginTime,
                     LusiTime::Time()) ;
}

void
ConnectionImpl::endRun (const std::string&    instrument,
                        const std::string&    experiment,
                        int                   run,
                        const LusiTime::Time& endTime) throw (WrongParams,
                                                              DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    if (!endTime.isValid ())
        throw WrongParams ("the begin run timstamp isn't valid") ;

    try {

        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, instrument, experiment))
            throw WrongParams ("unknown experiment") ;

        // Find the run in the database and make sure it's still open, and its begin time
        // is strictly less than the specified end time.
        //
        RunDescr run_descr ;
        if (!this->findRun (run_descr, exper_descr.id, run))
            throw WrongParams ("no such run in the database") ;


        if (run_descr.end_time.isValid())
            throw WrongParams ("the run is already ended") ;

        if (run_descr.begin_time >= endTime)
            throw WrongParams ("the specified end time isn't newer than the begin time of the run") ;

        // Now proceed with the new run creation
        //
        std::ostringstream sql;
        sql << "UPDATE run SET end_time=" << LusiTime::Time::to64 (endTime)
            << " WHERE id=" << run_descr.id ;

        this->simpleQuery (m_logbook_mysql, sql.str());

    } catch (const LusiTime::Exception& e) {
        throw WrongParams (
            std::string ("failed to translate LusiTime::Time to string because of: ")
            + e.what()) ;
    }
}

void
ConnectionImpl::saveFiles (const std::string& instrument,
                           const std::string& experiment,
                           int                run,
                           const std::string& run_type) throw (WrongParams,
                                                               DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    std::string type = run_type ;
    string2upper (type) ;
    if (!LogBook::isValidRunType (type))
        throw WrongParams ("unknown run type: "+run_type) ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    // Now proceed with the new data set creation
    //
    std::ostringstream sql;
    sql << "INSERT INTO fileset VALUES(NULL,NULL,"
        << "(SELECT id FROM fileset_status_def WHERE name='Initial_Entry')"
        << ",'" << experiment
        << "','" << instrument << "','" << type << "'," << run << ",NULL,NULL,NOW(),0)";

    this->simpleQuery (m_ifacedb_mysql, sql.str());
}

 void
 ConnectionImpl::saveFiles (const std::string& instrument,
                            const std::string& experiment,
                            int                run,
                            const std::string& run_type,
                            const std::vector<std::string >& files,
                            const std::vector<std::string >& file_types) throw (WrongParams,
                                                                                DatabaseError)
{
    if ( files.size() != file_types.size())
        throw WrongParams ("'files' length does not match the one of the 'file_types'") ;

    std::vector<std::string > types;
    types.reserve (file_types.size()) ;
    for (size_t i = 0; i < file_types.size(); i++) {

        std::string type = file_types[i] ;
        string2upper (type) ;
        if ( !isValidFileType (type))
            throw WrongParams ("unsupported file type: "+file_types[i]) ;

        types.push_back (type) ;
    }

    // Create initial data set
    //
    this->saveFiles( instrument, experiment, run, run_type) ;

    // Obtain the data set ID
    //
    QueryProcessor query (m_ifacedb_mysql) ;
    query.execute ("SELECT LAST_INSERT_ID() AS 'id'") ;
    if (!query.next_row())
        throw DatabaseError ("inconsistent result from the database") ;

    int id = 0 ;
    query.get (id, "id") ;

    // Register files with the data set
    //
    for (size_t i = 0; i < files.size(); i++) {
        std::ostringstream sql;
        sql << "INSERT INTO files VALUES(NULL," << id << ","
            << "(SELECT id FROM fileset_status_def WHERE name='Waiting_Translation'),'" << files[i]
            << "','" << types[i] << "',NULL)";

        this->simpleQuery (m_ifacedb_mysql, sql.str());
    }

    // Change data set status to make it ready for processing
    //
    std::ostringstream sql;
    sql << "UPDATE fileset SET fk_fileset_status=(SELECT id FROM fileset_status_def WHERE name='Waiting_Translation')"
        << " WHERE id=" << id;

    this->simpleQuery (m_ifacedb_mysql, sql.str());
}

void
ConnectionImpl::createRunParam (const std::string& instrument,
                                const std::string& experiment,
                                const std::string& parameter,
                                const std::string& parameter_type,
                                const std::string& description) throw (WrongParams,
                                                                        DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    std::string type = parameter_type ;
    string2upper (type) ;
    if (!isValidValueType(type))
        throw WrongParams ("unsupported run parameter type: "+parameter_type) ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    // Now proceed with the new run parameter creation
    //
    std::ostringstream sql;
    sql << "INSERT INTO run_param VALUES(NULL,'"
        << this->escape_string (m_logbook_mysql, parameter) << "',"
        << exper_descr.id << ",'"
        << type << "','"
        << this->escape_string (m_logbook_mysql, description) << "')";

    this->simpleQuery (m_logbook_mysql, sql.str());
}

void
ConnectionImpl::setRunParam (const std::string& instrument,
                             const std::string& experiment,
                             int                run,
                             const std::string& parameter,
                             int                value,
                             const std::string& source,
                             bool               updateAllowed) throw (ValueTypeMismatch,
                                                                      WrongParams,
                                                                      DatabaseError)
{
    this->setRunParamImpl (instrument,
                           experiment,
                           run,
                           parameter,
                           value,
                           "INT",
                           source,
                           updateAllowed) ;
}

void
ConnectionImpl::setRunParam (const std::string& instrument,
                             const std::string& experiment,
                             int                run,
                             const std::string& parameter,
                             double             value,
                             const std::string& source,
                             bool               updateAllowed) throw (ValueTypeMismatch,
                                                                      WrongParams,
                                                                      DatabaseError)
{
    this->setRunParamImpl (instrument,
                           experiment,
                           run,
                           parameter,
                           value,
                           "DOUBLE",
                           source,
                           updateAllowed) ;
}

void
ConnectionImpl::setRunParam (const std::string& instrument,
                             const std::string& experiment,
                             int                run,
                             const std::string& parameter,
                             const std::string& value,
                             const std::string& source,
                             bool               updateAllowed) throw (ValueTypeMismatch,
                                                                      WrongParams,
                                                                      DatabaseError)
{
    this->setRunParamImpl<std::string > (instrument,
                                         experiment,
                                         run,
                                         parameter,
                                         "'"+this->escape_string (m_logbook_mysql, value)+"'",
                                         "TEXT",
                                         source,
                                         updateAllowed) ;
}

void
ConnectionImpl::getRunParam (const std::string& instrument,
                             const std::string& experiment,
                             int                run,
                             const std::string& parameter,
                             int&               value,
                             std::string&       source,
                             LusiTime::Time&    updated) throw (ValueTypeMismatch,
                                                                WrongParams,
                                                                DatabaseError)
{
    QueryProcessor query (m_logbook_mysql) ;
    this->getRunParamImpl (query,
                           instrument,
                           experiment,
                           run,
                           parameter,
                           "INT") ;

    query.get (value,   "val") ;
    query.get (source,  "source") ;
    query.get (updated, "updated") ;
}

void
ConnectionImpl::getRunParam (const std::string& instrument,
                             const std::string& experiment,
                             int                run,
                             const std::string& parameter,
                             double&            value,
                             std::string&       source,
                             LusiTime::Time&    updated) throw (ValueTypeMismatch,
                                                                WrongParams,
                                                                DatabaseError)
{
    QueryProcessor query (m_logbook_mysql) ;
    this->getRunParamImpl (query,
                           instrument,
                           experiment,
                           run,
                           parameter,
                           "DOUBLE") ;

    query.get (value,   "val") ;
    query.get (source,  "source") ;
    query.get (updated, "updated") ;
}

void
ConnectionImpl::getRunParam (const std::string& instrument,
                             const std::string& experiment,
                             int                run,
                             const std::string& parameter,
                             std::string&       value,
                             std::string&       source,
                             LusiTime::Time&    updated) throw (ValueTypeMismatch,
                                                                WrongParams,
                                                                DatabaseError)
{
    QueryProcessor query (m_logbook_mysql) ;
    this->getRunParamImpl (query,
                           instrument,
                           experiment,
                           run,
                           parameter,
                           "TEXT") ;

    query.get (value,   "val") ;
    query.get (source,  "source") ;
    query.get (updated, "updated") ;
}

void
ConnectionImpl::reportOpenFile (int exper_id,
                                int run,
                                int stream,
                                int chunk,
                                const std::string& host,
                                const std::string& dirpath,
                                const std::string& scope) throw (WrongParams,
                                                                 DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // Make sure both the experiment and the run are already known
    //
    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, exper_id))
        throw WrongParams ("unknown experiment") ;

    LusiTime::Time run_request_time ;
    if (!this->findRunRequest (run_request_time, exper_id, run))
        throw WrongParams ("unknown run for the experiment") ;

    // The current timestamp will be recorded as a time when the new file opening
    // was registered.
    //
    const LusiTime::Time now = LusiTime::Time::now () ;

    // Host name and directory path shouldn't exceed the limit. Neither they should
    // be empty.
    //
    if(host.empty() || (host.size() > 255) || dirpath.empty() || (dirpath.size() > 255))
        throw WrongParams ("host name or directory path are either empty or exceed the limit of 255 characters") ;

    // Now proceed with the new file registration
    //

    /* The operation requires the following table (note that table name may
     * be longer in case if non-empty <scope> is specified):

       CREATE TABLE `file[_<scope>]` (
         `exper_id` int(11) NOT NULL,
         `run`      int(11) NOT NULL,
         `stream`   int(11) NOT NULL,
         `chunk`    int(11) NOT NULL,
         `open`     bigint(20) unsigned NOT NULL,
         `host`     varchar(255) NOT NULL,
         `dirpath`  varchar(255) NOT NULL,
          PRIMARY KEY  (`exper_id`, `run`, `stream`, `chunk`),
          KEY `FILE_FK_1` (`exper_id`),
          CONSTRAINT `FILE_FK_1` FOREIGN KEY (`exper_id`) REFERENCES `experiment` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
       ) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_general_cs;

    */

    std::ostringstream sql;
    sql << "INSERT INTO file" << (scope.empty() ? "" : "_" + scope) << " VALUES("
        << exper_id << ","
        << run      << ","
        << stream   << ","
        << chunk    << ","
        << LusiTime::Time::to64 (now) << ",'"
        << this->escape_string (m_regdb_mysql, host)    << "','"
        << this->escape_string (m_regdb_mysql, dirpath) << "')";

    /*
    std::cout << "DEBUG: ConnectionImpl::reportOpenFile: sql=" << sql.str() << std::endl;
    return;
    */
    this->simpleQuery (m_regdb_mysql, sql.str());
}

bool
ConnectionImpl::getAttrInfo (AttrInfo&          info,
                             const std::string& instrument,
                             const std::string& experiment,
                             int                run,
                             const std::string& attr_class,
                             const std::string& attr_name) throw (WrongParams,
                                                                  DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream sql;
    sql << "SELECT * FROM run_attr WHERE run_id=" << run_descr.id
        << " AND class='" << attr_class << "' AND name='" << attr_name << "'";

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    if (query.num_rows() > 1)
        throw DatabaseError ("inconsistent result returned by query - database may be corrupt") ;

    while (query.next_row()) {
        LogBook::row2attr(info, query, instrument, experiment, run);
        return true ;
    }
    return false ;
}

void
ConnectionImpl::getAttrInfo (std::vector<AttrInfo >& info,
                             const std::string&      instrument,
                             const std::string&      experiment,
                             int                     run,
                             const std::string&      attr_class) throw (WrongParams,
                                                                        DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream sql;
    sql << "SELECT * FROM run_attr WHERE run_id=" << run_descr.id
        << " AND class='" << attr_class << "' ORDER BY name";

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    info.reserve (query.num_rows ()) ;

    while (query.next_row()) {

        AttrInfo param ;

        LogBook::row2attr(param, query, instrument, experiment, run);

        info.push_back (param) ;
    }
}

void
ConnectionImpl::getAttrInfo (std::vector<AttrInfo >& info,
                             const std::string&      instrument,
                             const std::string&      experiment,
                             int                     run) throw (WrongParams,
                                                                 DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream sql;
    sql << "SELECT * FROM run_attr WHERE run_id=" << run_descr.id << " ORDER BY class, name";

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    info.reserve (query.num_rows ()) ;

    while (query.next_row()) {

        AttrInfo param ;

        LogBook::row2attr(param, query, instrument, experiment, run);

        info.push_back (param) ;
    }
}

void
ConnectionImpl::getAttrClasses (std::vector<std::string >& attr_classes,
                                const std::string&         instrument,
                                const std::string&         experiment,
                                int                        run) throw (WrongParams,
                                                                       DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream sql;
    sql << "SELECT DISTINCT class FROM run_attr WHERE run_id=" << run_descr.id << " ORDER BY class";

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    attr_classes.reserve (query.num_rows ()) ;

    while (query.next_row()) {

        std::string attr_class ;
        query.get (attr_class, "class") ;
        attr_classes.push_back (attr_class) ;
    }
}

bool
ConnectionImpl::getAttrVal (long&              attr_value,
                            const std::string& instrument,
                            const std::string& experiment,
                            int                run,
                            const std::string& attr_class,
                            const std::string& attr_name) throw (ValueTypeMismatch,
                                                                 WrongParams,
                                                                 DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream sql;
    sql << "SELECT run_attr_int.val AS `val` FROM run_attr, run_attr_int WHERE run_attr.run_id=" << run_descr.id
        << " AND run_attr.class='" << attr_class << "'"
        << " AND run_attr.id=run_attr_int.attr_id";

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    if (query.num_rows () > 1)
        throw DatabaseError ("inconsistent result returned by query - database may be corrupt") ;

    while (query.next_row()) {
        query.get (attr_value, "val") ;
        return true ;
    }
    return false ;
}

bool
ConnectionImpl::getAttrVal (double&            attr_value,
                            const std::string& instrument,
                            const std::string& experiment,
                            int                run,
                            const std::string& attr_class,
                            const std::string& attr_name) throw (ValueTypeMismatch,
                                                                 WrongParams,
                                                                 DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream sql;
    sql << "SELECT run_attr_double.val AS `val` FROM run_attr, run_attr_double WHERE run_attr.run_id=" << run_descr.id
        << " AND run_attr.class='" << attr_class << "'"
        << " AND run_attr.id=run_attr_double.attr_id";

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    if (query.num_rows () > 1)
        throw DatabaseError ("inconsistent result returned by query - database may be corrupt") ;

    while (query.next_row()) {
        query.get (attr_value, "val") ;
        return true ;
    }
    return false ;
}

bool
ConnectionImpl::getAttrVal (std::string&       attr_value,
                            const std::string& instrument,
                            const std::string& experiment,
                            int                run,
                            const std::string& attr_class,
                            const std::string& attr_name) throw (ValueTypeMismatch,
                                                                 WrongParams,
                                                                 DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream sql;
    sql << "SELECT run_attr_text.val AS `val` FROM run_attr, run_attr_text WHERE run_attr.run_id=" << run_descr.id
        << " AND run_attr.class='" << attr_class << "'"
        << " AND run_attr.id=run_attr_text.attr_id";

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    if (query.num_rows () > 1)
        throw DatabaseError ("inconsistent result returned by query - database may be corrupt") ;

    while (query.next_row()) {
        query.get (attr_value, "val") ;
        return true ;
    }
    return false ;
}

void
ConnectionImpl::createRunAttr (const std::string& instrument,
                               const std::string& experiment,
                               int                run,
                               const std::string& attr_class,
                               const std::string& attr_name,
                               const std::string& attr_description,
                               long               attr_value) throw (WrongParams,
                                                                     DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    std::string attr_type = "INT" ;

    if( attr_name.empty())
        throw WrongParams ("attribute name can't be empty") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream s_1, s_2;
    s_1 << "INSERT INTO run_attr VALUES(NULL,"
        << run_descr.id << ",'"
        << this->escape_string (m_logbook_mysql, attr_class) << "','"
        << this->escape_string (m_logbook_mysql, attr_name) << "','"
        << attr_type << "','"
        << this->escape_string (m_logbook_mysql, attr_description) << "')";
    s_2 << "INSERT INTO run_attr_int VALUES(LAST_INSERT_ID(),"
        << attr_value << ");";

    this->simpleQuery (m_logbook_mysql, s_1.str());
    this->simpleQuery (m_logbook_mysql, s_2.str());
}

void
ConnectionImpl::createRunAttr (const std::string& instrument,
                               const std::string& experiment,
                               int                run,
                               const std::string& attr_class,
                               const std::string& attr_name,
                               const std::string& attr_description,
                               double             attr_value) throw (WrongParams,
                                                                     DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    std::string attr_type = "DOUBLE" ;

    if( attr_name.empty())
        throw WrongParams ("attribute name can't be empty") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream s_1, s_2;
    s_1 << "INSERT INTO run_attr VALUES(NULL,"
        << run_descr.id << ",'"
        << this->escape_string (m_logbook_mysql, attr_class) << "','"
        << this->escape_string (m_logbook_mysql, attr_name) << "','"
        << attr_type << "','"
        << this->escape_string (m_logbook_mysql, attr_description) << "');";
    s_2 << "INSERT INTO run_attr_double VALUES(LAST_INSERT_ID(),"
        << attr_value << ");";

    this->simpleQuery (m_logbook_mysql, s_1.str());
    this->simpleQuery (m_logbook_mysql, s_2.str());
}

void
ConnectionImpl::createRunAttr (const std::string& instrument,
                               const std::string& experiment,
                               int                run,
                               const std::string& attr_class,
                               const std::string& attr_name,
                               const std::string& attr_description,
                               const std::string& attr_value) throw (WrongParams,
                                                                     DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    std::string attr_type = "TEXT" ;

    if( attr_name.empty())
        throw WrongParams ("attribute name can't be empty") ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    std::ostringstream s_1, s_2;
    s_1 << "INSERT INTO run_attr VALUES(NULL,"
        << run_descr.id << ",'"
        << this->escape_string (m_logbook_mysql, attr_class) << "','"
        << this->escape_string (m_logbook_mysql, attr_name) << "','"
        << attr_type << "','"
        << this->escape_string (m_logbook_mysql, attr_description) << "');";
    s_2 << "INSERT INTO run_attr_text VALUES(LAST_INSERT_ID(),'"
        <<  this->escape_string (m_logbook_mysql, attr_value) << "');";

    this->simpleQuery (m_logbook_mysql, s_1.str());
    this->simpleQuery (m_logbook_mysql, s_2.str());
}

bool
ConnectionImpl::findExper (ExperDescr&        descr,
                           const std::string& instrument,
                           const std::string& experiment) throw (WrongParams,
                                                                 DatabaseError)
{
    std::ostringstream sql;
    sql << "SELECT i.name AS 'instr_name',i.descr AS 'instr_descr',e.* FROM "
        << "instrument i, experiment e WHERE i.name='" << instrument
        << "' AND e.name='" << experiment
        << "' AND e.instr_id=i.id" ;

    return this->findExperImpl( descr, sql.str()) ;
}

bool
ConnectionImpl::findExper (ExperDescr& descr,
                           int         exper_id) throw (WrongParams,
                                                        DatabaseError)
{
    std::ostringstream sql;
    sql << "SELECT i.name AS 'instr_name',i.descr AS 'instr_descr',e.* FROM "
        << "instrument i, experiment e WHERE e.id=" << exper_id << " AND e.instr_id=i.id" ;

    return this->findExperImpl( descr, sql.str()) ;
}

bool
ConnectionImpl::findExperImpl (ExperDescr&        descr,
                               const std::string& sql) throw (WrongParams,
                                                              DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // Execute the query
    //
    QueryProcessor query (m_regdb_mysql) ;
    query.execute (sql) ;

    // Extract results
    //
    if (!query.next_row()) return false ;

    query.get (descr.instr_id,    "instr_id") ;
    query.get (descr.instr_name,  "instr_name") ;
    query.get (descr.instr_descr, "instr_descr") ;

    query.get (descr.id,    "id") ;
    query.get (descr.name,  "name") ;
    query.get (descr.descr, "descr") ;

    query.get (descr.registration_time, "registration_time") ;
    query.get (descr.begin_time,        "begin_time") ;
    query.get (descr.end_time,          "end_time") ;

    query.get (descr.leader_account, "leader_account") ;
    query.get (descr.contact_info,   "contact_info") ;
    query.get (descr.posix_gid,      "posix_gid") ;

    return true ;
}

bool
ConnectionImpl::findRunRequest (LusiTime::Time& request_time,
                                int             exper_id,
                                int             run) throw (WrongParams,
                                                            DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT request_time FROM run_" << exper_id << " WHERE num=" << run ;

    QueryProcessor query (m_regdb_mysql) ;
    query.execute (sql.str()) ;

    // Extract results
    //
    if (!query.next_row()) return false ;

    query.get (request_time, "request_time") ;

    return true ;
}

bool
ConnectionImpl::findRunParam (ParamDescr&        descr,
                              int                exper_id,
                              const std::string& name) throw (WrongParams,
                                                              DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT * FROM run_param WHERE exper_id=" << exper_id
        << " AND param='" << name << "'" ;

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    // Extract results
    //
    if (!query.next_row()) return false ;

    query.get (descr.id,       "id") ;
    query.get (descr.name,     "param") ;
    query.get (descr.exper_id, "exper_id") ;
    query.get (descr.type,     "type") ;
    query.get (descr.descr,    "descr", true) ;

    return true ;
}

bool
ConnectionImpl::findRun (RunDescr& descr,
                         int       exper_id,
                         int       num) throw (WrongParams,
                                               DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT * FROM run WHERE exper_id=" << exper_id
        << " AND num=" << num ;

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    // Extract results
    //
    if (!query.next_row()) return false ;

    query.get (descr.id,         "id") ;
    query.get (descr.num,        "num") ;
    query.get (descr.exper_id,   "exper_id") ;
    query.get (descr.type,       "type") ;
    query.get (descr.begin_time, "begin_time") ;
    query.get (descr.end_time,   "end_time", true) ;

    return true ;
}

bool
ConnectionImpl::findLastRun (RunDescr& descr,
                             int       exper_id) throw (WrongParams,
                                                        DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT * FROM run WHERE exper_id=" << exper_id
        << " AND begin_time=(SELECT MAX(begin_time) FROM run WHERE exper_id=" << exper_id << ")";

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    // Extract results
    //
    if (!query.next_row()) return false ;

    query.get (descr.id,         "id") ;
    query.get (descr.num,        "num") ;
    query.get (descr.exper_id,   "exper_id") ;
    query.get (descr.type,       "type") ;
    query.get (descr.begin_time, "begin_time") ;
    query.get (descr.end_time,   "end_time", true) ;

    return true ;
}

bool
ConnectionImpl::runParamValueIsSet (int param_id,
                                    int run_id) throw (WrongParams,
                                                       DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT * FROM run_val WHERE param_id=" << param_id
        << " AND run_id=" << run_id ;

    QueryProcessor query (m_logbook_mysql) ;
    query.execute (sql.str()) ;

    // Extract results (true if there is a row)
    //
    return query.next_row() ;
}

void
ConnectionImpl::getRunParamImpl (QueryProcessor&    query,
                                 const std::string& instrument,
                                 const std::string& experiment,
                                 int                run,
                                 const std::string& parameter,
                                 const std::string& parameter_type) throw (ValueTypeMismatch,
                                                                           WrongParams,
                                                                           DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    std::string type = parameter_type ;
    string2upper (type) ;
    if (!isValidValueType(type))
        throw WrongParams ("unsupported run parameter type: "+parameter_type) ;

    ExperDescr exper_descr ;
    if (!this->findExper (exper_descr, instrument, experiment))
        throw WrongParams ("unknown experiment") ;

    ParamDescr param_descr ;
    if (!findRunParam (param_descr,
                       exper_descr.id,
                       parameter))
        throw WrongParams ("no such parameter exists" );

    if (type != param_descr.type)
        throw ValueTypeMismatch ("actual arameter type is: "+param_descr.type+", not: "+type) ;

    RunDescr run_descr ;
    if (!findRun (run_descr,
                  exper_descr.id,
                  run))
        throw WrongParams ("unknown run") ;

    // Formulate and execute the query
    //
    std::ostringstream sql;
    sql << "SELECT v.source, v.updated, vv.val FROM run_val v, run_val_" << type << " vv"
        << " WHERE v.param_id=" << param_descr.id << " AND vv.param_id=" << param_descr.id
        << " AND   v.run_id="   << run_descr.id   << " AND vv.run_id="   << run_descr.id;

    query.execute (sql.str()) ;
    if (!query.next_row())
        throw WrongParams ("the value of the parameter isn't set yet for this run") ;
}

void
ConnectionImpl::simpleQuery (MYSQL* mysql, const std::string& query) throw (DatabaseError)
{
    if (mysql_real_query (mysql, query.c_str(), query.size()))
        throw DatabaseError( std::string( "error in mysql_real_query('"+query+"'): " ) + mysql_error(mysql));
}

std::string
ConnectionImpl::escape_string (MYSQL* mysql, const std::string& in_str) const throw ()
{
    const size_t in_str_len = in_str.length () ;
    std::auto_ptr<char > out_str (new char [2*in_str_len + 1]) ;
    const size_t out_str_len =
        mysql_real_escape_string (
            mysql,
            out_str.get(),
            in_str.c_str(),
            in_str_len) ;

    return std::string (out_str.get(), out_str_len) ;
}

} // namespace LogBook
