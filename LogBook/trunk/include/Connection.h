#ifndef LOGBOOK_CONNECTION_H
#define LOGBOOK_CONNECTION_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	    $Id: $
//
// Description:
//	    Class Connection. The top-level class of the "LogBook Database" API.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "LogBook/ValueTypeMismatch.h"
#include "LogBook/WrongParams.h"
#include "LogBook/DatabaseError.h"

#include "LusiTime/Time.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace LogBook {

/**
  * Connection parameters information structure.
  */
struct ConnectionParams {
    std::string host ;          // database host name
    std::string user ;          // database user account name
    bool        using_password ;// a flag indicating of a pasword iss required to connect
    std::string logbook ;       // database name for LogBook
    std::string regdb ;         // database name for RegDB
} ;

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
  * Parameter information structure.
  *
  * Data members of the class represent properties of the parameter
  * in the database.
  */
struct ParamInfo {
    std::string instrument ;    // its instrument name
    std::string experiment ;    // its experiment name
    std::string name ;          // its (parameter's) name
    std::string type ;          // its type
    std::string descr ;         // its description (can be very long!)
} ;

std::ostream& operator<< (std::ostream& s, const ParamInfo& p) ;

/**
  * The top-level class of the "LogBook Database" API.
  *
  * Note, that this class is made abstract to prevent clients code
  * from having compile-time dependency onto the implementation details
  * of the API.
  *
  * User application begin interacting with the API by obtaining an
  * instance of the class.
  *
  * This software was developed for the LCLS project.  If you use all or 
  * part of it, please give an appropriate acknowledgement.
  *
  * @version $Id$
  *
  * @author Igor Gaponenko
  */
class Connection  {

public:

    /**
      * Destructor.
      *
      * All open database connection will be closed. Note that no automatic
      * commits will be assumed in that case. A calling applicaton should
      * explicitly manage transactions by using the "beginTransaction()" and
      * "endTransaction()" methods declared below.
      */
    virtual ~Connection () throw () ;

    // Selectors (const)

    /**
      * Get connection parameters.
      */
    virtual ConnectionParams connParams () const = 0 ;

    // ------------------------------------------------------------
    // ------------- Transaction management methods ---------------
    // ------------------------------------------------------------

    /**
      * Begin the transaction.
      *
      * It's safe to call the method more than once during a session.
      * No nested transactions are supported.
      */
    virtual void beginTransaction () throw (DatabaseError) = 0;

    /**
      * Commit the transaction.
      *
      * Once the previous transaction is commited there won't be any
      * outstanding transaction.
      */
    virtual void commitTransaction () throw (DatabaseError) = 0;

    /**
      * Abort the current transaction (if any).
      *
      * All the modifications made since the (first non-commited or
      * non-aborted back) call to the "beginTransaction" will be lost.
      */
    virtual void abortTransaction () throw (DatabaseError) = 0;

    /**
      * Check the status of the current transaction.
      *
      * Return 'true' if the transaction is active.
      */
    virtual bool transactionIsStarted () const = 0;

    // ------------------------------------------------------------------
    // ------------- Database contents inspection methods ---------------
    // ------------------------------------------------------------------

    /**
      * Find experiment descriptors in the given scope.
      *
      * If the "instrument" parameter is given some non-default value
      * then experiments for the specified instrument will be searched for.
      * Otherwise all experiments will be returned.
      */
    virtual void getExperiments (std::vector<ExperDescr >& experiments,
                                 const std::string&        instrument="") throw (WrongParams,
                                                                                 DatabaseError) = 0 ;

    /**
      * Find an information on a parameter.
      *
      * The method would check if the parameter is known in the database
      * for the specified run. If so then hthe method will return 'true'
      * and set up a contents of the structure. Otherwise it will return
      * 'false'.
      *
      * EXCEPTIONS:
      *
      *   "WrongParams"   : to report wrong parameters (non-existing instrument, experiment, etc.)
      *   "DatabaseError" : to report database related problems
      *
      * @see class WrongParams
      * @see class DatabaseError
      */
     virtual bool getParamInfo (ParamInfo&         info,
                                const std::string& instrument,
                                const std::string& experiment,
                                const std::string& parameter) throw (WrongParams,
                                                                      DatabaseError) = 0 ;
    /**
      * Find an information on all parameters of an experiment.
      *
      * EXCEPTIONS:
      *
      *   "WrongParams"   : to report wrong parameters (non-existing instrument, experiment, etc.)
      *   "DatabaseError" : to report database related problems
      *
      * @see class WrongParams
      * @see class DatabaseError
      */
     virtual void getParamsInfo (std::vector<ParamInfo >& info,
                                 const std::string&       instrument,
                                 const std::string&       experiment) throw (WrongParams,
                                                                            DatabaseError) = 0 ;

     /**
      * Get a value of a run parameter (integer value).
      *
      * The method would get a value of the parameter in a scope of its
      * experiment and run. The run is given by its number. A type of
      * of the parameter should match the type defined in the experiment's
      * configuration.
      *
      * EXCEPTIONS:
      *
      *   "ValueTypeMismatch"  : to report a wrong type of a parameter's value
      *   "WrongParams"        : to report wrong parameters (non-existing experiment, etc.)
      *   "DatabaseError"      : to report database related problems
      *
      * @see class ValueTypeMismatch
      * @see class WrongParams
      * @see class DatabaseError
      */
    virtual void getRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              int&               value,
                              std::string&       source,
                              LusiTime::Time&    updated) throw (ValueTypeMismatch,
                                                                 WrongParams,
                                                                 DatabaseError) = 0 ;

    /**
      * Get a value of a run parameter (double value).
      */
    virtual void getRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              double&            value,
                              std::string&       source,
                              LusiTime::Time&    updated) throw (ValueTypeMismatch,
                                                                 WrongParams,
                                                                 DatabaseError) = 0 ;

     /**
      * Get a value of a run parameter (string value).
      */
    virtual void getRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              std::string&       value,
                              std::string&       source,
                              LusiTime::Time&    updated) throw (ValueTypeMismatch,
                                                                 WrongParams,
                                                                 DatabaseError) = 0 ;

    // --------------------------------------------------------------------
    // ------------- Database contents modification methods ---------------
    // --------------------------------------------------------------------

    /** Allocate next run number
      */
     virtual int allocateRunNumber (const std::string& instrument,
                                    const std::string& experiment) throw (WrongParams,
                                                                          DatabaseError) = 0 ;

    /**
      * Create a placeholder for a new run.
      *
      * The method would create a new run entry in the database. The run
      * will be assigned the specified number and it will have the range.
      *
      * NOTE #1: The method will end the previously open (if any) run.
      *
      * NOTE #2: The method should be used when the end time of the run is
      *          not known.
      *
      * EXCEPTIONS:
      *
      *   "WrongParams"   : to report wrong parameters (non-existing experiment, etc.)
      *   "DatabaseError" : to report database related problems
      *
      * @see class WrongParams
      * @see class DatabaseError
      */
     virtual void createRun (const std::string&    instrument,
                             const std::string&    experiment,
                             int                   run,
                             const std::string&    type,
                             const LusiTime::Time& beginTime,
                             const LusiTime::Time& endTime) throw (WrongParams,
                                                                   DatabaseError) = 0 ;
     /**
      * Begin a new run by a placeholder for the run.
      *
      * The method would create a new run entry in the database. The run
      * will be assigned the specified number and it will stay open untill
      * it's closed either explicitly (by calling the endRun() method) or
      * implicitly (by calling either createRun() or beginRun() method).
      *
      * NOTE #1: The method will end the previously open (if any) run.
      *
      * NOTE #2: The method should be used when the end time of the run is
      *          not known.
      *
      * EXCEPTIONS:
      *
      *   "WrongParams"   : to report wrong parameters (non-existing experiment, etc.)
      *   "DatabaseError" : to report database related problems
      *
      * @see class WrongParams
      * @see class DatabaseError
      */
     virtual void beginRun (const std::string&    instrument,
                            const std::string&    experiment,
                            int                   run,
                            const std::string&    type,
                            const LusiTime::Time& beginTime) throw (WrongParams,
                                                                    DatabaseError) = 0 ;

     /**
      * End a previously started run.
      *
      * The method would close an open-enede run in the database.
      *
      * EXCEPTIONS:
      *
      *   "WrongParams"   : to report wrong parameters (non-existing experiment, etc.)
      *   "DatabaseError" : to report database related problems
      *
      * @see class WrongParams
      * @see class DatabaseError
      */
     virtual void endRun (const std::string&    instrument,
                          const std::string&    experiment,
                          int                   run,
                          const LusiTime::Time& endTime) throw (WrongParams,
                                                                DatabaseError) = 0 ;

    /**
      * Create new run parameter.
      *
      * The method would define a new parameter for runs of an experiments.
      * Values of the parameters will have to be set separatedly for each run
      * by calling the corresponding "setRunParam()" metghod defined below.
      *
      * PARAMETERS:
      *
      *   parameter   - is a unique name of the parameter in a scope of an experiment
      *   type        - is a type of the parameter. Allowed values are: 'INT', 'DOUBLE'
      *                 or 'TEXT'.
      *   description - an arbitrary description of the parameter.
      *
      * EXCEPTIONS:
      *
      *   "WrongParams"        : to report wrong parameters (non-existing experiment, etc.)
      *   "DatabaseError"      : to report database related problems
      *
      * @see class ValueTypeMismatch
      * @see class WrongParams
      * @see class DatabaseError
      */
    virtual void createRunParam (const std::string& instrument,
                                 const std::string& experiment,
                                 const std::string& parameter,
                                 const std::string& type,
                                 const std::string& description) throw (WrongParams,
                                                                        DatabaseError) = 0 ;

 
     /**
      * Set a value of a run parameter (integer value).
      *
      * The method would set a value of the parameter in a scope of its
      * experiment and run. The run is given by its number. A type of
      * of the parameter should match the type defined in the experiment's
      * configuration.
      *
      * EXCEPTIONS:
      *
      *   "ValueTypeMismatch"  : to report a wrong type of a parameter's value
      *   "WrongParams"        : to report wrong parameters (non-existing experiment, etc.)
      *   "DatabaseError"      : to report database related problems
      *
      * @see class ValueTypeMismatch
      * @see class WrongParams
      * @see class DatabaseError
      */
    virtual void setRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              int                value,
                              const std::string& source,
                              bool               updateAllowed=false) throw (ValueTypeMismatch,
                                                                             WrongParams,
                                                                             DatabaseError) = 0 ;

    /**
      * Set a value of a run parameter (double value).
      */
    virtual void setRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              double             value,
                              const std::string& source,
                              bool               updateAllowed=false) throw (ValueTypeMismatch,
                                                                             WrongParams,
                                                                             DatabaseError) = 0 ;
     /**
      * Set a value of a run parameter (string value).
      */
    virtual void setRunParam (const std::string& instrument,
                              const std::string& experiment,
                              int                run,
                              const std::string& parameter,
                              const std::string& value,
                              const std::string& source,
                              bool               updateAllowed=false) throw (ValueTypeMismatch,
                                                                             WrongParams,
                                                                             DatabaseError) = 0 ;

protected:

    /**
      * Default constructor.
      */
    Connection () ;

private:

    // Copy constructor and assignment are disabled

    Connection ( const Connection& ) ;
    Connection& operator = ( const Connection& ) ;

    // Helper methods, implementations, etc.

//------------------
// Static Members --
//------------------

public:

    /**
      * Establish the connection.
      *
      * The connection will be open using the specified  connection parameters.
      * In case of a success, a valid pointer is returned. The object's ownership
      * is also retured by the method. So, don't forget to delete the object!
      *
      * EXCEPTIONS:
      *
      *   "DatabaseError" : to report database related problems
      *
      * @see class DatabaseError
      */
    static Connection* open ( const char* host     = 0,         // default: localhost
                              const char* user     = 0,         // default: UNIX user account
                              const char* password = 0,         // default: no password
                              const char* logbook  = "LogBook",
                              const char* regdb    = "RegDB" ) throw (DatabaseError) ;

    // Selectors (const)

    // Modifiers

private:

    // Data members

    static Connection* s_conn ; // cached connection (if any)
} ;

} // namespace LogBook

#endif // LOGBOOK_CONNECTION_H
