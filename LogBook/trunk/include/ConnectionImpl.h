#ifndef LOGBOOK_CONNECTIONIMPL_H
#define LOGBOOK_CONNECTIONIMPL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	    $Id: $
//
// Description:
//	    Class ConnectionImpl. An implementation of the "LogBook Database"
//      connector.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
#include <iostream>
#include <sstream>
#include <map>

#include <mysql/mysql.h>

//----------------------
// Base Class Headers --
//----------------------

#include "LogBook/Connection.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "LusiTime/Time.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace LogBook {

/**
  * Experiment descriptor.
  *
  * Data members of the class represent properties of the experiment
  * in the database.
  */
struct ExperDescr {

    // Instrument identity
    //
    int         instr_id ;
    std::string instr_name ;
    std::string instr_descr ;

    // Experiment identity
    //
    int         id ;
    std::string name ;
    std::string descr ;

    // Experimenttime frame
    //
    LusiTime::Time registration_time ;
    LusiTime::Time begin_time ;
    LusiTime::Time end_time ;

    // Experiment owner
    //
    std::string leader_account ;
    std::string contact_info ;
    std::string posix_gid ;
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
  * This is the final implementation of the "LogBook Database" connector.
  *
  * The class extends the abstract base class Connection by implementing
  * its methods. The class is bot meant to be used directly by API users.
  *
  * This software was developed for the LCLS project.  If you use all or 
  * part of it, please give an appropriate acknowledgement.
  *
  * @see class Connection
  *
  * @version $Id: $
  *
  * @author Igor Gaponenko
  */
class ConnectionImpl : public Connection  {

public:

    /**
      * Normal constructor.
      *
      * Take over the MySQL connection structure. Use it to make queries
      * and dispose it when destroying this object.
      */
    ConnectionImpl (MYSQL* mysql, const ConnectionParams& conn_params ) ;

    /**
      * Destructor.
      *
      * @see method Connection::~Connection()
      */
    virtual ~ConnectionImpl () throw () ;

    /**
      * Get connection parameters.
      *
      * @see method Connection::connParams()
      */
    virtual ConnectionParams connParams () const 
    {
        return m_conn_params ;
    }

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
     virtual bool getParamInfo (ParamInfo&         info,
                                const std::string& instrument,
                                const std::string& experiment,
                                const std::string& parameter) throw (WrongParams,
                                                                     DatabaseError) ;

    /**
      * Allocate next run number
      *
      * @see method Connection::allocateRunNumber()
      */
     virtual int allocateRunNumber (const std::string& instrument,
                                    const std::string& experiment) throw (DatabaseError) ;

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
      * Begin a new run by a placeholder for the run.
      *
      * @see method Connection::beginRun()
      */
     virtual void beginRun (const std::string&    instrument,
                            const std::string&    experiment,
                            int                   run,
                            const std::string&    type,
                            const LusiTime::Time& beginTime) throw (WrongParams,
                                                                    DatabaseError) ;

     /**
      * End a previously started run.
      *
      * @see method Connection::endRun()
      */
     virtual void endRun (const std::string&    instrument,
                          const std::string&    experiment,
                          int                   run,
                          const LusiTime::Time& beginTime) throw (WrongParams,
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
            throw WrongParams ("unknown experiment: " + experiment) ;
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
            throw WrongParams ("unknown experiment: " + experiment) ;
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

    /**
      * Execure a simple query which won't produce any result set.
      */
    void simpleQuery (const std::string& query) throw (DatabaseError) ;

private:

    // Data members

    bool m_is_started ;     // there is an active transaction

    MYSQL* m_mysql ;

    ConnectionParams m_conn_params ;
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
    if (updating) {

        std::ostringstream sql_1;
        sql_1 << "UPDATE " << m_conn_params.logbook << ".run_val"
              << " SET source='" << source << "', updated=NOW() WHERE run_id=" << runDescr.id
              << " AND param_id=" << paramDescr.id ;
        this->simpleQuery (sql_1.str()) ;

        std::ostringstream sql_2;
        sql_2 << "UPDATE " << m_conn_params.logbook << ".run_val_" << type
              << " SET val='" << value << "' WHERE run_id=" << runDescr.id
              << " AND param_id=" << paramDescr.id;
        this->simpleQuery (sql_2.str()) ;

    } else {

        std::ostringstream sql_1;
        sql_1 << "INSERT INTO " << m_conn_params.logbook << ".run_val"
              << " VALUES(" << runDescr.id << "," << paramDescr.id << ",'" << source << "', NOW())" ;
        this->simpleQuery (sql_1.str()) ;

        std::ostringstream sql_2;
        sql_2 << "INSERT INTO " << m_conn_params.logbook << ".run_val_" << type
              << " VALUES(" << runDescr.id << "," << paramDescr.id << ",'" << value << "')" ;
        this->simpleQuery (sql_2.str()) ;
    }
}

} // namespace LogBook

#endif // LOGBOOK_CONNECTIONIMPL_H
