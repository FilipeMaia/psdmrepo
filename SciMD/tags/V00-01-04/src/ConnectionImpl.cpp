//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
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
#include "SciMD/ConnectionImpl.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <memory>
#include <iostream>

#include <strings.h>

using namespace std ;

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "SciMD/ConnectionImpl.h"

#include "odbcpp/OdbcStatement.h"
#include "odbcpp/OdbcParam.h"
#include "odbcpp/OdbcResult.h"

#include "LusiTime/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace odbcpp ;

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace SciMD {

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

bool
isValidParamType(const std::string& type)
{
    static const char* const validTypes[] = {"INT", "INT64", "DOUBLE", "TEXT" } ;
    static const size_t numTypes = 4 ;
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
    s << "SciMD::ExperDescr {\n"
      << "          id: " << d.id << "\n"
      << "        name: " << d.name << "\n"
      << "    instr_id: " << d.instr_id << "\n"
      << "  begin_time: " << d.begin_time << "\n"
      << "    end_time: " << d.end_time << "\n"
      << "       descr: " << d.descr << "\n"
      << "}\n" ;
    return s ;
}

std::ostream&
operator<< (std::ostream& s, const ParamDescr& d)
{
    s << "SciMD::ParamDescr {\n"
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
    s << "SciMD::RunDescr {\n"
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

ConnectionImpl::ConnectionImpl (const odbcpp::OdbcConnection& odbc_conn_scimd,
                                const odbcpp::OdbcConnection& odbc_conn_regdb) :
    Connection () ,
    m_is_started (false) ,
    m_odbc_conn_scimd (odbc_conn_scimd),
    m_odbc_conn_regdb (odbc_conn_regdb)
{}

//--------------
// Destructor --
//--------------

ConnectionImpl::~ConnectionImpl () throw ()
{}

//-----------
// Methods --
//-----------

std::string
ConnectionImpl::connStringSciMD () const
{
    return m_odbc_conn_scimd.connString () ;
}

std::string
ConnectionImpl::connStringRegDB () const
{
    return m_odbc_conn_regdb.connString () ;
}

void
ConnectionImpl::beginTransaction () throw (DatabaseError)
{
    if (m_is_started) return ;

    try {
        m_odbc_conn_scimd.statement ("BEGIN").execute() ;
        m_odbc_conn_regdb.statement ("BEGIN").execute() ;
    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
    m_is_started = true ;
}

void
ConnectionImpl::commitTransaction () throw (DatabaseError)
{
    if (!m_is_started) return ;
    try {
        m_odbc_conn_scimd.statement ("COMMIT").execute() ;
        m_odbc_conn_regdb.statement ("COMMIT").execute() ;
    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
    m_is_started = false ;
}

void
ConnectionImpl::abortTransaction () throw (DatabaseError)
{
    if (!m_is_started) return ;
    try {
        m_odbc_conn_scimd.statement ("ROLLBACK").execute();
        m_odbc_conn_regdb.statement ("ROLLBACK").execute();
    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
    m_is_started = false ;
}

bool
ConnectionImpl::transactionIsStarted () const
{
    return m_is_started ;
}

bool
ConnectionImpl::getParamInfo (ParamInfo& info,
                              const std::string& instrument,
                              const std::string& experiment,
                              const std::string& parameter) throw (WrongParams,
                                                                   DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    try {
        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, instrument, experiment))
            throw WrongParams ("unknown experiment") ;

        ParamDescr param_descr ;
        if (!this->findRunParam (param_descr, exper_descr.id, parameter))
            return false ;

        info.experiment = instrument ;
        info.experiment = experiment ;
        info.name       = parameter ;
        info.type       = param_descr.type ;
        info.descr      = param_descr.descr ;

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
    return true ;
}

void
ConnectionImpl::defineParam (const std::string& instrument,
                             const std::string& experiment,
                             const std::string& parameter,
                             const std::string& type,
                             const std::string& description) throw (WrongParams,
                                                                    DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    // TODO: Consider reinforcing the types at a level of the API interface
    //       (by using 'enum' rather than here.
    //
    if (!SciMD::isValidParamType (type))
        throw WrongParams ("unknown type of the parameter") ;

    try {

        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, instrument, experiment))
            throw WrongParams ("unknown experiment") ;

        OdbcStatement stmt = m_odbc_conn_scimd.statement (
            "INSERT INTO run_param VALUES(NULL,?,?,?,?)");

        OdbcParam<std::string> p1 (parameter) ;
        OdbcParam<int>         p2 (exper_descr.id) ;
        OdbcParam<std::string> p3 (type) ;
        OdbcParam<std::string> p4 (description) ;

        stmt.bindParam (1, p1) ;
        stmt.bindParam (2, p2) ;
        stmt.bindParam (3, p3) ;
        stmt.bindParam (4, p4) ;

        OdbcResultPtr result = stmt.execute() ;

        stmt.unbindParams() ;

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
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

    if (!endTime.isValid ())
        throw WrongParams ("the begin run timstamp isn't valid") ;

    // TODO: Consider reinforcing the types at a level of the API interface
    //       (by using 'enum' rather than here.
    //
    if (!SciMD::isValidRunType (type))
        throw WrongParams ("unknown run type") ;

    try {
        long long unsigned beginTime64 = LusiTime::Time::to64 (beginTime) ;
        long long unsigned   endTime64 = LusiTime::Time::to64 (  endTime) ;

        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, instrument, experiment))
            throw WrongParams ("unknown experiment") ;

        OdbcStatement stmt = m_odbc_conn_scimd.statement (
            "INSERT INTO run VALUES(NULL,?,?,?,?,?)");

        OdbcParam<int>         p1 (run) ;
        OdbcParam<int>         p2 (exper_descr.id) ;
        OdbcParam<std::string> p3 (type) ;
        OdbcParam<long long unsigned> p4 (beginTime64) ;
        OdbcParam<long long unsigned> p5 (endTime64) ;

        stmt.bindParam (1, p1) ;
        stmt.bindParam (2, p2) ;
        stmt.bindParam (3, p3) ;
        stmt.bindParam (4, p4) ;
        stmt.bindParam (5, p5) ;

        OdbcResultPtr result = stmt.execute() ;

        stmt.unbindParams() ;

    } catch (const LusiTime::Exception& e) {
        throw WrongParams (
            std::string ("failed to translate LusiTime::Time to string because of: ")
            + e.what()) ;

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
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
                             int64_t            value,
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
                                  "INT64",
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

    try {

        // Formulate and make the query
        //
        OdbcStatement stmt = m_odbc_conn_regdb.statement ("SELECT e.*,i.name as instr_name FROM instrument i, experiment e WHERE i.name=? AND e.name=? AND i.id=e.instr_id") ;

        OdbcParam<std::string> p_instrument (instrument) ;
        OdbcParam<std::string> p_experiment (experiment) ;

        stmt.bindParam (1, p_instrument) ;
        stmt.bindParam (2, p_experiment) ;

        OdbcResultPtr result = stmt.execute() ;

        stmt.unbindParams() ;

        // Extract results
        //
        if (result->empty()) return false ;

        OdbcColumnVar<int >         col_id  ;
        OdbcColumnVar<std::string > col_name  (255) ;
        OdbcColumnVar<int >         col_instr_id ;
        OdbcColumnVar<std::string > col_instr_name (255) ;
        OdbcColumnVar<std::string > col_begin_time (255) ;
        OdbcColumnVar<std::string > col_end_time   (255) ;
        OdbcColumnVar<std::string > col_descr      (255) ;

        result->bindColumn ("id",         col_id) ;
        result->bindColumn ("name",       col_name) ;
        result->bindColumn ("instr_id",   col_instr_id) ;
        result->bindColumn ("instr_name", col_instr_name) ;
        result->bindColumn ("begin_time", col_begin_time) ;
        result->bindColumn ("end_time",   col_end_time) ;
        result->bindColumn ("descr",      col_descr) ;

        const unsigned int nRows = result->fetch() ;
        if (!nRows) return false ;
        if (nRows != 1)
            throw DatabaseError ("database contents corrupted, exactly one record expected") ;

        const unsigned int row = 0 ;

        descr.id         = col_id.value         (row) ;
        descr.name       = col_name.value       (row) ;
        descr.instr_id   = col_instr_id.value   (row) ;
        descr.instr_name = col_instr_name.value (row) ;
        descr.begin_time = col_begin_time.value (row) ;
        if (!col_end_time.isNull()) descr.end_time = col_end_time.value (row) ;
        if (!col_descr.isNull())    descr.descr    = col_descr.value    (row) ;

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
    return true ;}

bool
ConnectionImpl::findRunParam (ParamDescr&        descr,
                              int                exper_id,
                              const std::string& name) throw (WrongParams,
                                                              DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    try {

        // Formulate and make the query
        //
        OdbcStatement stmt = m_odbc_conn_scimd.statement ("SELECT * FROM run_param WHERE exper_id = ? AND param = ?") ;

        OdbcParam<int>         p1 (exper_id) ;
        OdbcParam<std::string> p2 (name) ;

        stmt.bindParam ( 1, p1) ;
        stmt.bindParam ( 2, p2) ;

        OdbcResultPtr result = stmt.execute() ;

        stmt.unbindParams() ;

        // Extract results
        //
        if (result->empty()) return false ;

        OdbcColumnVar<int >         col_id ;
        OdbcColumnVar<std::string > col_name  (255) ;
        OdbcColumnVar<int >         col_exper_id  ;
        OdbcColumnVar<std::string > col_type  (255) ;
        OdbcColumnVar<std::string > col_descr (256*256) ;

        result->bindColumn ("id",       col_id) ;
        result->bindColumn ("param",    col_name) ;
        result->bindColumn ("exper_id", col_exper_id) ;
        result->bindColumn ("type",     col_type) ;
        result->bindColumn ("descr",    col_descr) ;

        const unsigned int nRows = result->fetch() ;
        if (!nRows) return false ;
        if (nRows != 1)
            throw DatabaseError ("database contents corrupted, exactly one record expected") ;

        const unsigned int row = 0 ;

        descr.id         = col_id.value       (row) ;
        descr.name       = col_name.value     (row) ;
        descr.exper_id   = col_exper_id.value (row) ;
        descr.type       = col_type.value     (row) ;
        if (!col_descr.isNull()) descr.descr  = col_descr.value (row) ;

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
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

    try {

        // Formulate and make the query
        //
        OdbcStatement stmt = m_odbc_conn_scimd.statement ("SELECT * FROM run WHERE exper_id = ? AND num = ?") ;

        OdbcParam<int > p1 (exper_id) ;
        OdbcParam<int > p2 (num) ;

        stmt.bindParam ( 1, p1) ;
        stmt.bindParam ( 2, p2) ;

        OdbcResultPtr result = stmt.execute() ;

        stmt.unbindParams() ;

        // Extract results
        //
        if (result->empty()) return false ;

        OdbcColumnVar<int >         col_id ;
        OdbcColumnVar<int >         col_num ;
        OdbcColumnVar<int >         col_exper_id ;
        OdbcColumnVar<std::string > col_type       (255) ;
        OdbcColumnVar<std::string > col_begin_time (255) ;
        OdbcColumnVar<std::string > col_end_time   (255) ;

        result->bindColumn ("id",         col_id) ;
        result->bindColumn ("num",        col_num) ;
        result->bindColumn ("exper_id",   col_exper_id) ;
        result->bindColumn ("type",       col_type) ;
        result->bindColumn ("begin_time", col_begin_time) ;
        result->bindColumn ("end_time",   col_end_time) ;

        const unsigned int nRows = result->fetch() ;
        if (!nRows) return false ;
        if (nRows != 1)
            throw DatabaseError ("database contents corrupted, exactly one record expected") ;

        const unsigned int row = 0 ;

        descr.id         = col_id.value         (row) ;
        descr.num        = col_num.value        (row) ;
        descr.exper_id   = col_exper_id.value   (row) ;
        descr.type       = col_type.value       (row) ;
        descr.begin_time = col_begin_time.value (row) ;
        descr.end_time   = col_end_time.value   (row) ;

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
    return true ;
}

bool
ConnectionImpl::runParamValueIsSet (int param_id,
                                    int run_id) throw (WrongParams,
                                                       DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    try {

        // Formulate and make the query
        //
        OdbcStatement stmt = m_odbc_conn_scimd.statement ("SELECT * FROM run_val WHERE param_id = ? AND run_id = ?") ;

        OdbcParam<int > p1 (param_id) ;
        OdbcParam<int > p2 (run_id) ;

        stmt.bindParam ( 1, p1) ;
        stmt.bindParam ( 2, p2) ;

        OdbcResultPtr result = stmt.execute() ;

        stmt.unbindParams() ;

        // Extract results
        //
        return result->fetch () > 0 ;

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
}

} // namespace SciMD
