#ifndef SCIMD_CONNECTIONIMPL_H
#define SCIMD_CONNECTIONIMPL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	    $Id$
//
// Description:
//	    Class ConnectionImpl. An implementation of the "Science Metadata
//      Database" connector.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
#include <ostream>

//----------------------
// Base Class Headers --
//----------------------

#include "SciMD/Connection.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "odbcpp/OdbcConnection.h"
#include "odbcpp/OdbcStatement.h"
#include "odbcpp/OdbcParam.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace SciMD {

/**
  * Experiment descriptor.
  *
  * Data members of the class represent properties of the experiment
  * in the database.
  */
struct ExperDescr {
    int         id ;
    std::string name ;
    int         instr_id ;
    std::string instr_name ;
    std::string begin_time ;
    std::string end_time ;
    std::string descr ;
} ;

std::ostream& operator<< (std::ostream& s, const ExperDescr& d) ;


/**
  * Parameter descriptor.
  *
  * Data members of the class represent properties of the parameter
  * in the database.
  */
struct ParamDescr {
    int         id ;
    std::string name ;
    int         exper_id ;
    std::string type ;
    std::string descr ;
} ;

std::ostream& operator<< (std::ostream& s, const ParamDescr& d) ;

/**
  * Run descriptor.
  *
  * Data members of the class represent properties of the experiment
  * in the database.
  */
struct RunDescr {
    int         id ;
    int         num ;
    int         exper_id ;
    std::string type ;
    std::string begin_time ;
    std::string end_time ;
} ;

std::ostream& operator<< (std::ostream& s, const RunDescr& d) ;

/**
  * This is the final implementation of the "Science Metadata Database" connector.
  *
  * The class extends the abstract base class Connection by implementing
  * its methods. The class is bot meant to be used directly by API users.
  *
  * This software was developed for the LUSI project.  If you use all or 
  * part of it, please give an appropriate acknowledgement.
  *
  * @ see class Connection
  *
  * @version $Id$
  *
  * @author Igor Gaponenko
  */
class ConnectionImpl : public Connection  {

public:

    /**
      * Normal constructor.
      *
      * Expects two ODBC connectors as input parameters - one for SciMD and
      * the other one for RegDB.
      */
    explicit ConnectionImpl (const odbcpp::OdbcConnection& odbc_conn_scimd,
                             const odbcpp::OdbcConnection& odbc_conn_regdb) ;

    /**
      * Destructor.
      *
      * @see method Connection::~Connection()
      */
    virtual ~ConnectionImpl () throw () ;

    /**
      * Get ODBC connection string for the SciMD database.
      *
      * @see method Connection::connStringSciMD()
      */
    virtual std::string connStringSciMD () const ;

    /**
      * Get ODBC connection string for the regDB database.
      */
    virtual std::string connStringRegDB () const ;

    /**
      * Begin the transaction.
      *
      * @see method Connection::beginTransaction()
      */
    virtual void beginTransaction () throw (DatabaseError);

    /**
      * Commit the transaction.
      *
      * @see method Connection::commitTransaction()
      */
    virtual void commitTransaction () throw (DatabaseError);

    /**
      * Abort the current transaction (if any).
      *
      * @see method Connection::abortTransaction()
      */
    virtual void abortTransaction () throw (DatabaseError);

    /**
      * Check the status of the current transaction.
      *
      * @see method Connection::transactionIsStarted()
      */
    virtual bool transactionIsStarted () const;

    /**
      * Find an information on a parameter.
      *
      * @see method Connection::getParamInfo()
      */
     virtual bool getParamInfo (ParamInfo& info,
                                const std::string& instrument,
                                const std::string& experiment,
                                const std::string& parameter) throw (WrongParams,
                                                                     DatabaseError) ;
    /**
      * Define a new run parameter for an experiment.
      *
      * @see method Connection::defineParam()
      */
     virtual void defineParam (const std::string& instrument,
                               const std::string& experiment,
                               const std::string& parameter,
                               const std::string& type,
                               const std::string& description) throw (WrongParams,
                                                                      DatabaseError) ;

    /**
      * Create a placeholder for a new run.
      *
      * @see method Connection::createRun()
      */
     virtual void createRun (const std::string&    instrument,
                             const std::string&    experiment,
                             int                   run,
                             const std::string&    type,
                             const LusiTime::Time& beginTime,
                             const LusiTime::Time& endTime) throw (WrongParams,
                                                                   DatabaseError) ;

    /**
      * Set a value of a run parameter (integer value).
      *
      * @see method Connection::setRunParam()
      */
    virtual void setRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              int                value,
                              const std::string& source,
                              bool               updateAllowed=false) throw (ValueTypeMismatch,
                                                                             WrongParams,
                                                                             DatabaseError) ;
    /**
      * Set a value of a run parameter (64-bit integer value).
      *
      * @see method Connection::setRunParam()
      */
    virtual void setRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              int64_t            value,
                              const std::string& source,
                              bool               updateAllowed=false) throw (ValueTypeMismatch,
                                                                             WrongParams,
                                                                             DatabaseError) ;

    /**
      * Set a value of a run parameter (double value).
      *
      * @see method Connection::setRunParam()
      */
    virtual void setRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              double             value,
                              const std::string& source,
                              bool               updateAllowed=false) throw (ValueTypeMismatch,
                                                                             WrongParams,
                                                                             DatabaseError) ;

    /**
      * Set a value of a run parameter (string value).
      *
      * @see method Connection::setRunParam()
      */
    virtual void setRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              const std::string& value,
                              const std::string& source,
                              bool               updateAllowed=false) throw (ValueTypeMismatch,
                                                                             WrongParams,
                                                                             DatabaseError) ;

private:

    // Default constructor is disabled

    ConnectionImpl () ;

    // Copy constructor and assignment are disabled

    ConnectionImpl ( const ConnectionImpl& ) ;
    ConnectionImpl& operator = ( const ConnectionImpl& ) ;

    // Helper methods, implementations, etc.

    /**
      * Find experiment description in the database.
      *
      * The method will return 'true' and it will set up values
      * of the supplied data structure if the parameter is found.
      */
    bool findExper(ExperDescr&        descr,
                   const std::string& instrument,
                   const std::string& experiment) throw (WrongParams,
                                                         DatabaseError) ;

    /**
      * Find run parameter description in the database.
      *
      * The method will return 'true' and it will set up values
      * of the supplied data structure if the parameter is found.
      */
    bool findRunParam (ParamDescr&        descr,
                       int                exper_id,
                       const std::string& parameter) throw (WrongParams,
                                                            DatabaseError) ;
    /**
      * Find run parameter description in the database.
      *
      * The method will return 'true' and it will set up values
      * of the supplied data structure if the parameter is found.
      */
    bool findRunParam (ParamDescr&        descr,
                       const std::string& instrument,
                       const std::string& experiment,
                       const std::string& parameter) throw (WrongParams,
                                                            DatabaseError)
    {
        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, instrument, experiment))
            throw WrongParams ("unknown instrument/experiment: " + instrument + "/" + experiment) ;
        return this->findRunParam (descr, exper_descr.id, parameter) ;
    }

    /**
      * Find run description in the database.
      *
      * The method will return 'true' and it will set up values
      * of the supplied data structure if the parameter is found.
      */
    bool findRun (RunDescr& descr,
                  int       exper_id,
                  int       num) throw (WrongParams,
                                        DatabaseError) ;

    /**
      * Find run description in the database.
      *
      * The method will return 'true' and it will set up values
      * of the supplied data structure if the parameter is found.
      */
    bool findRun (RunDescr&          descr,
                  const std::string& instrument,
                  const std::string& experiment,
                  int                num) throw (WrongParams,
                                                 DatabaseError)
    {
        ExperDescr exper_descr ;
        if (!this->findExper (exper_descr, instrument, experiment))
            throw WrongParams ("unknown instrument/experiment: " + instrument + "/" + experiment) ;
        return this->findRun (descr, exper_descr.id, num) ;
    }

    /**
      * Check if a value of the run parameter has already been set for the run.
      *
      * The method will return 'true' if the value is already set.
      */
    bool runParamValueIsSet (int param_id,
                             int run_id) throw (WrongParams,
                                                DatabaseError) ;

    /**
      * The actual implementation of the operation for all types of parameter
      * values.
      *
      * The function is used by the class's methods.
      */
    template <class T >
    void setRunParamImpl (const std::string& instrument,
                          const std::string& experiment,
                          int                run,
                          const std::string& parameter,
                          T                  value,
                          const std::string& type,
                          const std::string& source,
                          bool               updateAllowed) throw (ValueTypeMismatch,
                                                                   WrongParams,
                                                                   DatabaseError) ;

private:

    // Data members

    bool m_is_started ;     // there is an active transaction

    odbcpp::OdbcConnection m_odbc_conn_scimd ;  // ODBC connector (SciMD)
    odbcpp::OdbcConnection m_odbc_conn_regdb ;  // ODBC connector (RegDB)
};

//------------------
// Implementation --
//------------------

template <class T >
void
ConnectionImpl::setRunParamImpl (const std::string& instrument,
                                 const std::string& experiment,
                                 int                run,
                                 const std::string& parameter,
                                 T                  value,
                                 const std::string& type,
                                 const std::string& source,
                                 bool               updateAllowed) throw (ValueTypeMismatch,
                                                                          WrongParams,
                                                                          DatabaseError)
{
    if (!m_is_started)
        throw DatabaseError ("no transaction") ;

    try {

        // Get the parameter from the database to make sure that the parameter
        // has been configured. Also figure out its type. We want to be sure
        // that the value is of the correct type.
        //
        RunDescr runDescr ;
        if (!findRun (runDescr, instrument, experiment, run))
            throw WrongParams ("unknown run") ;

        ParamDescr paramDescr ;
        if (!findRunParam (paramDescr, instrument, experiment, parameter))
            throw WrongParams ("unknown parameter") ;

        if (paramDescr.type != type)
            throw ValueTypeMismatch ("unexpected parameter type") ;

        // Check if the parameter's value already exists in the database.
        // If so, and if the update is allowed - then do rthe update
        // instead of insertion.
        //
        bool updating = false ;
        if (this->runParamValueIsSet (paramDescr.id, runDescr.id)) {
            if (updateAllowed)
                updating = true ;
            else
                throw DatabaseError ("the value of parameter is alreay set") ;
        }

        // Proceed with the operation and insert/update an entry into two tables.
        //
        long long unsigned setOrUpdateTime64 = LusiTime::Time::to64 (LusiTime::Time::now()) ;

        odbcpp::OdbcParam<int >               p_run_id   (runDescr.id) ;
        odbcpp::OdbcParam<int >               p_param_id (paramDescr.id) ;
        odbcpp::OdbcParam<std::string>        p_source   (source) ;
        odbcpp::OdbcParam<T>                  p_value    (value) ;
        odbcpp::OdbcParam<long long unsigned> p_updated  (setOrUpdateTime64) ;

        if (updating) {
            {
                odbcpp::OdbcStatement stmt = m_odbc_conn_scimd.statement (
                    "UPDATE run_val SET source=?, updated=? WHERE run_id=? AND param_id=?");

                stmt.bindParam (1, p_source) ;
                stmt.bindParam (2, p_updated) ;
                stmt.bindParam (3, p_run_id) ;
                stmt.bindParam (4, p_param_id) ;

                odbcpp::OdbcResultPtr result = stmt.execute() ;
                stmt.unbindParams() ;
            }
            {
                odbcpp::OdbcStatement stmt = m_odbc_conn_scimd.statement (
                    "UPDATE run_val_" + type + " SET val=? WHERE run_id=? AND param_id=?");

                stmt.bindParam (1, p_value) ;
                stmt.bindParam (2, p_run_id) ;
                stmt.bindParam (3, p_param_id) ;

                odbcpp::OdbcResultPtr result = stmt.execute() ;
                stmt.unbindParams() ;
            }
        } else {
            {
                odbcpp::OdbcStatement stmt = m_odbc_conn_scimd.statement (
                    "INSERT INTO run_val VALUES(?,?,?,?)");

                stmt.bindParam (1, p_run_id) ;
                stmt.bindParam (2, p_param_id) ;
                stmt.bindParam (3, p_source) ;
                stmt.bindParam (4, p_updated) ;

                odbcpp::OdbcResultPtr result = stmt.execute() ;
                stmt.unbindParams() ;
            }
            {
                odbcpp::OdbcStatement stmt = m_odbc_conn_scimd.statement (
                    "INSERT INTO run_val_" + type + " VALUES(?,?,?)");

                stmt.bindParam (1, p_run_id) ;
                stmt.bindParam (2, p_param_id) ;
                stmt.bindParam (3, p_value) ;

                odbcpp::OdbcResultPtr result = stmt.execute() ;
                stmt.unbindParams() ;
            }
        }

    } catch (const odbcpp::OdbcException& e) {
        throw DatabaseError (e.what()) ;
    }
}

} // namespace SciMD

#endif // SCIMD_CONNECTIONIMPL_H
