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

//---------------------------------------------------
// Collaborating Class Headers (fr this namespace) --
//---------------------------------------------------

class QueryProcessor ;

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
    int            id ;
    int            num ;
    int            exper_id ;
    std::string    type ;
    LusiTime::Time begin_time ;
    LusiTime::Time end_time ;
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
      * Take over the MySQL connection structures. Use them to make queries
      * and dispose them when destroying this object.
      */
    ConnectionImpl (MYSQL* logbook_mysql,
                    MYSQL*   regdb_mysql,
                    MYSQL* ifacedb_mysql) ;

    /**
      * Destructor.
      *
      * @see method Connection::~Connection()
      */
    virtual ~ConnectionImpl () throw () ;

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
      * Find the current experiment for the specified instrument and a station number
      *
      * @see method Connection::getCurrentExperiment()
      */
    virtual bool getCurrentExperiment (ExperDescr&        descr,
                                       const std::string& instrument,
                                       unsigned int       station=0) throw (WrongParams,
                                                                            DatabaseError) ;

    /**
      * Find experiment descriptors in the given scope.
      *
      * @see method Connection::getExperiments()
      */
    virtual void getExperiments (std::vector<ExperDescr >& experiments,
                                 const std::string&        instrument="") throw (WrongParams,
                                                                                 DatabaseError) ;

    /**
      * Find experiment descriptor of the specified experiment if the one exists
      *
      * @see method Connection::getOneExperiment()
      */
    virtual bool getOneExperiment (ExperDescr&        descr,
                                   const std::string& instrument,
                                   const std::string& experiment) throw (WrongParams,
                                                                         DatabaseError) ;

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
      * Find an information on all parameters of an experiment.
      *
      * @see method Connection::getParamsInfo()
      */
     virtual void getParamsInfo (std::vector<ParamInfo >& info,
                                 const std::string&       instrument,
                                 const std::string&       experiment) throw (WrongParams,
                                                                             DatabaseError) ;

     /**
      * Get a value of a run parameter (integer value).
      *
      * @see method Connection::getRunParam()
      */
    virtual void getRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              int                &value,
                              std::string&       source,
                              LusiTime::Time&    updated) throw (ValueTypeMismatch,
                                                                 WrongParams,
                                                                 DatabaseError) ;

    /**
      * Get a value of a run parameter (double value).
      *
      * @see method Connection::getRunParam()
      */
    virtual void getRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              double&            value,
                              std::string&       source,
                              LusiTime::Time&    updated) throw (ValueTypeMismatch,
                                                                 WrongParams,
                                                                 DatabaseError) ;

     /**
      * Get a value of a run parameter (string value).
      *
      * @see method Connection::getRunParam()
      */
    virtual void getRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              std::string&       value,
                              std::string&       source,
                              LusiTime::Time&    updated) throw (ValueTypeMismatch,
                                                                 WrongParams,
                                                                 DatabaseError) ;

    /**
      * Find an information on a specific run attribute.
      *
      * @see method Connection::getAttrInfo()
      */
     virtual bool getAttrInfo (AttrInfo&          info,
                               const std::string& instrument,
                               const std::string& experiment,
                               int                run,
                               const std::string& attr_class,
                               const std::string& attr_name) throw (WrongParams,
                                                                    DatabaseError) ;
    /**
      * Find an information on all run attributes of a given class.
      *
      * @see method Connection::getAttrInfo()
      */
     virtual void getAttrInfo (std::vector<AttrInfo >& info,
                               const std::string&      instrument,
                               const std::string&      experiment,
                               int                     run,
                               const std::string&      attr_class) throw (WrongParams,
                                                                          DatabaseError) ;


    /**
      * Find an information on all known run attributes.
      *
      * @see method Connection::getAttrInfo()
      */
     virtual void getAttrInfo (std::vector<AttrInfo >& info,
                               const std::string&      instrument,
                               const std::string&      experiment,
                               int                     run) throw (WrongParams,
                                                                   DatabaseError) ;

    /**
      * Find names of all known classes of run attributes.
      *
      * @see method Connection::getAttrClasses()
      */
     virtual void getAttrClasses (std::vector<std::string >& attr_classes,
                                  const std::string&         instrument,
                                  const std::string&         experiment,
                                  int                        run) throw (WrongParams,
                                                                         DatabaseError) ;

    /**
      * Obtain a value of a specific run attribute (32-bit integer).
      *
      * @see method Connection::getAttrVal()
      */
     virtual bool getAttrVal (long&              attr_value,
                              const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& attr_class,
                              const std::string& attr_name) throw (ValueTypeMismatch,
                                                                   WrongParams,
                                                                   DatabaseError) ;

    /**
      * Obtain a value of a specific run attribute (double).
      *
      * @see method Connection::getAttrVal()
      */
     virtual bool getAttrVal (double&            attr_value,
                              const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& attr_class,
                              const std::string& attr_name) throw (ValueTypeMismatch,
                                                                   WrongParams,
                                                                   DatabaseError) ;

    /**
      * Obtain a value of a specific run attribute (string).
      *
      * @see method Connection::getAttrVal()
      */
     virtual bool getAttrVal (std::string&       attr_value,
                              const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& attr_class,
                              const std::string& attr_name) throw (ValueTypeMismatch,
                                                                   WrongParams,
                                                                   DatabaseError) ;


    /**
      * Allocate next run number
      *
      * @see method Connection::allocateRunNumber()
      */
     virtual int allocateRunNumber (const std::string& instrument,
                                    const std::string& experiment) throw (WrongParams,
                                                                          DatabaseError) ;

    /**
      * Create a placeholder for a new run.
      *
      * @see method Connection::createRun()
      */
     virtual void createRun (const std::string&    instrument,
                             const std::string&    experiment,
                             int                   run,
                             const std::string&    run_type,
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
                            const std::string&    run_type,
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
                          const LusiTime::Time& endTime) throw (WrongParams,
                                                                DatabaseError) ;

     /**
      * Tell OFFLINE to save files for a run (OFFLINE will search for files).
      *
      * @see method Connection::saveFiles()
      */
     virtual void saveFiles (const std::string& instrument,
                             const std::string& experiment,
                             int                run,
                             const std::string& run_type) throw (WrongParams,
                                                                 DatabaseError) ;
     /**
      * Tell OFFLINE to save files for a run (expect an explicit list of files).
      */
     virtual void saveFiles (const std::string& instrument,
                             const std::string& experiment,
                             int                run,
                             const std::string& run_type,
                             const std::vector<std::string >& files,
                             const std::vector<std::string >& file_types) throw (WrongParams,
                                                                                 DatabaseError) ;

    /**
      * Create new run parameter.
      *
      * @see method Connection::createRunParam()
      */
    virtual void createRunParam (const std::string& instrument,
                                 const std::string& experiment,
                                 const std::string& parameter,
                                 const std::string& parameter_type,
                                 const std::string& description) throw (WrongParams,
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

    /**
      * Report an open file.
      *
      * @see method Connection::reportOpenFile()
      */
    virtual void reportOpenFile (int exper_id,
                                 int run,
                                 int stream,
                                 int chunk,
                                 const std::string& host,
                                 const std::string& dirpath,
                                 const std::string& scope=std::string()) throw (WrongParams,
                                                                                DatabaseError) ;

   /**
      * Create a new attribute of a run (integer value)
      *
      * @see method Connection::createRunAttr()
      */
    virtual void createRunAttr (const std::string& instrument,
                                const std::string& experiment,
                                int                run,
                                const std::string& attr_class,
                                const std::string& attr_name,
                                const std::string& attr_description,
                                long               attr_value) throw (WrongParams,
                                                                      DatabaseError) ;

    /**
      * Create a new attribute of a run (double precision floating point value)
      *
      * @see method Connection::createRunAttr()
      */
    virtual void createRunAttr (const std::string& instrument,
                                const std::string& experiment,
                                int                run,
                                const std::string& attr_class,
                                const std::string& attr_name,
                                const std::string& attr_description,
                                double             attr_value) throw (WrongParams,
                                                                      DatabaseError) ;

    /**
      * Create a new attribute of a run (text value)
      *
      * @see method Connection::createRunAttr()
      */
    virtual void createRunAttr (const std::string& instrument,
                                const std::string& experiment,
                                int                run,
                                const std::string& attr_class,
                                const std::string& attr_name,
                                const std::string& attr_description,
                                const std::string& attr_value) throw (WrongParams,
                                                                      DatabaseError) ;

private:

    // Default constructor is disabled

    ConnectionImpl () ;

    // Copy constructor and assignment are disabled

    ConnectionImpl ( const ConnectionImpl& ) ;
    ConnectionImpl& operator = ( const ConnectionImpl& ) ;

    // Helper methods, implementations, etc.

    /**
      * Find the current experiment identifier for the specified instrument and a station number
      *
      */
    bool getCurrentExperimentId (int&               id,
                                 const std::string& instrument,
                                 unsigned int       station) throw (WrongParams,
                                                                    DatabaseError) ;

    /**
      * Find experiment description in the database using instrument and experiment names.
      *
      * The method will return 'true' and it will set up values
      * of the supplied data structure if the experiemnt is found.
      */
    bool findExper(ExperDescr&        descr,
                   const std::string& instrument,
                   const std::string& experiment) throw (WrongParams,
                                                         DatabaseError) ;

    /**
      * Find experiment description in the database using experiment identifier.
      *
      * The method will return 'true' and it will set up values
      * of the supplied data structure if the experiment is found.
      */
    bool findExper(ExperDescr& descr,
                   int         exper_id ) throw (WrongParams,
                                                 DatabaseError) ;

    /**
      * Internal implementation of the experiment locator.
      *
      * The method will return 'true' and it will set up values
      * of the supplied data structure if the experiment is found.
      *
      * NOTE: This operation is used by two above defined experiment
      * locator methods.
      */
    bool findExperImpl(ExperDescr&        descr,
                       const std::string& sql) throw (WrongParams,
                                                      DatabaseError) ;

    /**
      * Find run number request time in the database.
      *
      * The method will return 'true' and it will set up a value
      * of the supplied data structure if the registration record is found.
      */
    bool findRunRequest (LusiTime::Time& request_time,
                         int             exper_id,
                         int             run) throw (WrongParams,
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
      * Find last run description in the database.
      *
      * The method will return 'true' and it will set up values
      * of the supplied data structure if the parameter is found.
      */
    bool findLastRun (RunDescr& descr,
                      int       exper_id) throw (WrongParams,
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
      * An actual "backend" implementation of the "getRunParam" methods.
      *
      * EXCEPTIONS:
      *
      *   "ValueTypeMismatch" : parameter's actual type won't match the proposed one
      *   "WrongParams"       : parameter value isn't set, no such run, experiment, etc.
      *   "DatabaseError"     : database errors, etc.
      *
      * The function is used by the class's methods.
      */
    void getRunParamImpl (QueryProcessor&    query,
                          const std::string& instrument,
                          const std::string& experiment,
                          int                run,
                          const std::string& parameter,
                          const std::string& parameter_type) throw (ValueTypeMismatch,
                                                                    WrongParams,
                                                                    DatabaseError) ;

    /**
      * The actual implementation of` the operation for all types of parameter
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
                          const std::string& parameter_type,
                          const std::string& source,
                          bool               updateAllowed) throw (ValueTypeMismatch,
                                                                   WrongParams,
                                                                   DatabaseError) ;

    /**
      * Execure a simple query which won't produce any result set.
      */
    void simpleQuery (MYSQL* mysql, const std::string& query) throw (DatabaseError) ;

    /**
      * A fron-end to mysql_real_escape_string()
      */
    std::string escape_string (MYSQL* mysql, const std::string& str) const throw () ;

private:

    // Data members

    bool m_is_started ;     // there is an active transaction

    MYSQL* m_logbook_mysql ;
    MYSQL* m_regdb_mysql ;
    MYSQL* m_ifacedb_mysql ;
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
                                 const std::string& parameter_type,
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

    if (paramDescr.type != parameter_type)
        throw ValueTypeMismatch ("unexpected parameter type: "+parameter_type) ;

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
    const LusiTime::Time now (LusiTime::Time::now ()) ;
    if (updating) {

        std::ostringstream sql_1;
        sql_1 << "UPDATE run_val"
              << " SET source='" << this->escape_string (m_logbook_mysql, source)
              << "', updated=" << LusiTime::Time::to64 (now) << " WHERE run_id=" << runDescr.id
              << " AND param_id=" << paramDescr.id ;
        this->simpleQuery (m_logbook_mysql, sql_1.str()) ;

        std::ostringstream sql_2;
        sql_2 << "UPDATE run_val_" << parameter_type
              << " SET val=" << value << " WHERE run_id=" << runDescr.id
              << " AND param_id=" << paramDescr.id;
        this->simpleQuery (m_logbook_mysql, sql_2.str()) ;

    } else {

        std::ostringstream sql_1;
        sql_1 << "INSERT INTO run_val"
              << " VALUES(" << runDescr.id << "," << paramDescr.id << ",'"
              << this->escape_string (m_logbook_mysql, source) << "', " << LusiTime::Time::to64 (now) << ")" ;
        this->simpleQuery (m_logbook_mysql, sql_1.str()) ;

        std::ostringstream sql_2;
        sql_2 << "INSERT INTO run_val_" << parameter_type
              << " VALUES(" << runDescr.id << "," << paramDescr.id << "," << value << ")" ;
        this->simpleQuery (m_logbook_mysql, sql_2.str()) ;
    }
}

} // namespace LogBook

#endif // LOGBOOK_CONNECTIONIMPL_H
