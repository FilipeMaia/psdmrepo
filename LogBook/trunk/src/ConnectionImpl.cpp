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

#include "Lusi/Lusi.h"

//-----------------------
// This Class's Header --
//-----------------------

#include "LogBook/ConnectionImpl.h"

//-----------------
// C/C++ Headers --
//-----------------

#include <strings.h>
#include <stdlib.h>

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

bool
isValidRunType(const std::string& type)
{
    static const char* const validTypes[] = {"DATA", "CALIB"} ;
    static const size_t numTypes = 2 ;
    for (size_t i = 0 ; i < numTypes ; ++i)
        if (0 == strcasecmp(validTypes[i], type.c_str()))
            return true ;
    return false ;
}

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

ConnectionImpl::ConnectionImpl (MYSQL* mysql, const ConnectionParams& conn_params) :
    Connection () ,
    m_is_started (false) ,
    m_mysql (mysql) ,
    m_conn_params (conn_params)
{}

//--------------
// Destructor --
//--------------

ConnectionImpl::~ConnectionImpl () throw ()
{
    mysql_close( m_mysql ) ;
    m_mysql = 0 ;
}

//-----------
// Methods --
//-----------

void
ConnectionImpl::beginTransaction () throw (DatabaseError)
{
    if (m_is_started) return ;
    this->simpleQuery ("BEGIN");
    m_is_started = true ;
}

void
ConnectionImpl::commitTransaction () throw (DatabaseError)
{
    if (!m_is_started) return ;
    this->simpleQuery ("COMMIT");
    m_is_started = false ;
}

void
ConnectionImpl::abortTransaction () throw (DatabaseError)
{
    if (!m_is_started) return ;
    this->simpleQuery ("ROLLBACK");
    m_is_started = false ;
}

bool
ConnectionImpl::transactionIsStarted () const
{
    return m_is_started ;
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
        sql << "INSERT INTO " << m_conn_params.regdb << ".run_" << exper_descr.id
            << " VALUES(NULL," << LusiTime::Time::to64 (now) << ")";

        this->simpleQuery (sql.str());

        // Get back its number
        //
        QueryProcessor query (m_mysql) ;
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
                           const std::string&    type,
                           const LusiTime::Time& beginTime,
                           const LusiTime::Time& endTime) throw (WrongParams,
                                                                 DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    if (!beginTime.isValid ())
        throw WrongParams ("the begin run timstamp isn't valid") ;

    // TODO: Consider reinforcing the types at a level of the API interface
    //       (by using 'enum' rather than here.
    //
    if (!LogBook::isValidRunType (type))
        throw WrongParams ("unknown run type") ;

    try {

        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, instrument, experiment))
            throw WrongParams ("unknown experiment") ;

        // Find out the previous run (if any) and make sure it gets closed
        // if it's still being open.
        //
        RunDescr run_descr ;
        if (this->findLastRun (run_descr, exper_descr.id)) {
            this->endRun (instrument, experiment, run_descr.num, beginTime) ;
        }

        // Now proceed with the new run creation
        //
        std::ostringstream sql;
        sql << "INSERT INTO " << m_conn_params.logbook << ".run VALUES(NULL,"
            << run << ","
            << exper_descr.id << ",'"
            << type << "',"
            << LusiTime::Time::to64 (beginTime) << ",";

        if (endTime.isValid ())
            sql << LusiTime::Time::to64 (endTime) << ")";
        else
            sql << "NULL)";

        this->simpleQuery (sql.str());

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
                          const std::string&    type,
                          const LusiTime::Time& beginTime) throw (WrongParams,
                                                                  DatabaseError)
{
    this->createRun (instrument,
                     experiment,
                     run,
                     type,
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
            throw WrongParams ("the specified run is already closed") ;

        if (run_descr.begin_time >= endTime)
            throw WrongParams ("the specified end time isn't newer than the begin time of the run") ;

        // Now proceed with the new run creation
        //
        std::ostringstream sql;
        sql << "UPDATE " << m_conn_params.logbook << ".run SET end_time=" << LusiTime::Time::to64 (endTime)
            << " WHERE id=" << run_descr.id ;

        this->simpleQuery (sql.str());

    } catch (const LusiTime::Exception& e) {
        throw WrongParams (
            std::string ("failed to translate LusiTime::Time to string because of: ")
            + e.what()) ;
    }
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
    return this->setRunParamImpl (instrument,
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
    return this->setRunParamImpl (instrument,
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
    return this->setRunParamImpl<std::string > (instrument,
                                                experiment,
                                                run,
                                                parameter,
                                                value,
                                                "TEXT",
                                                source,
                                                updateAllowed) ;
}

bool
ConnectionImpl::findExper (ExperDescr&        descr,
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
        << m_conn_params.regdb << ".instrument i, "
        << m_conn_params.regdb << ".experiment e WHERE i.name='" << instrument
        << "' AND e.name='" << experiment
        << "' AND e.instr_id=i.id" ;

    QueryProcessor query (m_mysql) ;
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
    sql << "SELECT * FROM " << m_conn_params.logbook << ".run_param "
        << " WHERE exper_id=" << exper_id
        << " AND param='" << name << "'" ;

    QueryProcessor query (m_mysql) ;
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
    sql << "SELECT * FROM " << m_conn_params.logbook << ".run "
        << " WHERE exper_id=" << exper_id
        << " AND num=" << num ;

    QueryProcessor query (m_mysql) ;
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
    sql << "SELECT * FROM " << m_conn_params.logbook << ".run "
        << " WHERE exper_id=" << exper_id
        << " AND begin_time=(SELECT MAX(begin_time) FROM " << m_conn_params.logbook << ".run "
        << " WHERE exper_id=" << exper_id << ")";

    QueryProcessor query (m_mysql) ;
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
    sql << "SELECT * FROM " << m_conn_params.logbook << ".run_val "
        << " WHERE param_id=" << param_id
        << " AND run_id=" << run_id ;

    QueryProcessor query (m_mysql) ;
    query.execute (sql.str()) ;

    // Extract results (true if there is a row)
    //
    return query.next_row() ;
}

void
ConnectionImpl::simpleQuery (const std::string& query) throw (DatabaseError)
{
    if (mysql_real_query (m_mysql, query.c_str(), query.size()))
        throw DatabaseError( std::string( "error in mysql_real_query('"+query+"'): " ) + mysql_error(m_mysql));
}

} // namespace LogBook
