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
      << "    group_id: " << d.group_id << "\n"
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

ConnectionImpl::ConnectionImpl (const odbcpp::OdbcConnection& odbc_conn) :
    Connection () ,
    m_is_started (false) ,
    m_odbc_conn (odbc_conn)
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
ConnectionImpl::connString () const
{
    return m_odbc_conn.connString () ;
}

void
ConnectionImpl::beginTransaction () throw (DatabaseError)
{
    if (m_is_started) return ;

    try {
        m_odbc_conn.statement ("BEGIN").execute() ;
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
        m_odbc_conn.statement ("COMMIT").execute() ;
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
        m_odbc_conn.statement ("ROLLBACK").execute();
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
                              const std::string& experiment,
                              const std::string& parameter) throw (WrongParams,
                                                                   DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    try {
        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, experiment))
            throw WrongParams ("unknown experiment") ;

        ParamDescr param_descr ;
        if (!this->findRunParam (param_descr, exper_descr.id, parameter))
            return false ;

        info.name       = parameter ;
        info.experiment = experiment ;
        info.type       = param_descr.type ;
        info.descr      = param_descr.descr ;

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
    return true ;
}

void
ConnectionImpl::createRun (const std::string&    experiment,
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

    try {
        const std::string beginTimeStr = beginTime.toString () ;
        const std::string   endTimeStr =   endTime.toString () ;

        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, experiment))
            throw WrongParams ("unknown experiment") ;

        OdbcStatement stmt = m_odbc_conn.statement (
            "INSERT INTO run VALUES(NULL,?,?,?,?,?)");

        OdbcParam<int>         p1 (run) ;
        OdbcParam<int>         p2 (exper_descr.id) ;
        OdbcParam<std::string> p3 (type) ;
        OdbcParam<std::string> p4 (beginTimeStr) ;
        OdbcParam<std::string> p5 (endTimeStr) ;

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
ConnectionImpl::setRunParam (const std::string& experiment,
                             int                run,
                             const std::string& parameter,
                             int                value,
                             const std::string& source,
                             bool               updateAllowed) throw (ValueTypeMismatch,
                                                                      WrongParams,
                                                                      DatabaseError)
{
    return this->setRunParamImpl (experiment,
                                  run,
                                  parameter,
                                  value,
                                  "INT",
                                  source,
                                  updateAllowed) ;
}

void
ConnectionImpl::setRunParam (const std::string& experiment,
                             int                run,
                             const std::string& parameter,
                             double             value,
                             const std::string& source,
                             bool               updateAllowed) throw (ValueTypeMismatch,
                                                                      WrongParams,
                                                                      DatabaseError)
{
    return this->setRunParamImpl (experiment,
                                  run,
                                  parameter,
                                  value,
                                  "DOUBLE",
                                  source,
                                  updateAllowed) ;
}

void
ConnectionImpl::setRunParam (const std::string& experiment,
                             int                run,
                             const std::string& parameter,
                             const std::string& value,
                             const std::string& source,
                             bool               updateAllowed) throw (ValueTypeMismatch,
                                                                      WrongParams,
                                                                      DatabaseError)
{
    return this->setRunParamImpl<std::string > (experiment,
                                                run,
                                                parameter,
                                                value,
                                                "TEXT",
                                                source,
                                                updateAllowed) ;
}

bool
ConnectionImpl::findExper (ExperDescr&        descr,
                           const std::string& name) throw (WrongParams,
                                                           DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    try {

        // Formulate and make the query
        //
        OdbcStatement stmt = m_odbc_conn.statement ("SELECT * FROM experiment WHERE name = ?") ;

        OdbcParam<std::string> p_name (name) ;

        stmt.bindParam ( 1, p_name) ;

        OdbcResultPtr result = stmt.execute() ;

        stmt.unbindParams() ;

        // Extract results
        //
        if (result->empty()) return false ;

        OdbcColumnVar<int >         col_id  ;
        OdbcColumnVar<std::string > col_name  (255) ;
        OdbcColumnVar<int >         col_instr_id ;
        OdbcColumnVar<int >         col_group_id ;
        OdbcColumnVar<std::string > col_begin_time (255) ;
        OdbcColumnVar<std::string > col_end_time   (255) ;
        OdbcColumnVar<std::string > col_descr      (255) ;

        result->bindColumn ("id",         col_id) ;
        result->bindColumn ("name",       col_name) ;
        result->bindColumn ("instr_id",   col_instr_id) ;
        result->bindColumn ("group_id",   col_group_id) ;
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
        descr.group_id   = col_group_id.value   (row) ;
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
        OdbcStatement stmt = m_odbc_conn.statement ("SELECT * FROM run_param WHERE exper_id = ? AND param = ?") ;

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
        OdbcStatement stmt = m_odbc_conn.statement ("SELECT * FROM run WHERE exper_id = ? AND num = ?") ;

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
        OdbcStatement stmt = m_odbc_conn.statement ("SELECT * FROM run_val WHERE param_id = ? AND run_id = ?") ;

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
