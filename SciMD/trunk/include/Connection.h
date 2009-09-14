#ifndef SCIMD_CONNECTION_H
#define SCIMD_CONNECTION_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	    $Id$
//
// Description:
//	    Class Connection. The top-level class of the "Science Metadata
//      Database" API.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "SciMD/ValueTypeMismatch.h"
#include "SciMD/WrongParams.h"
#include "SciMD/DatabaseError.h"

#include "LusiTime/Time.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace SciMD {

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

/**
  * The top-level class of the "Science Metadata Database" API.
  *
  * Note, that this class is made abstract to prevent clients code
  * from having compile-time dependency onto the implementation details
  * of the API.
  *
  * User application begin interacting with the API by obtaining an
  * instance of the class.
  *
  * This software was developed for the LUSI project.  If you use all or 
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
      * Get ODBC connection string for the database.
      */
    virtual std::string connString () const = 0 ;

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
      * Find an information on a parameter.
      *
      * The method would check if the parameter is known in the database
      * for the specified run. If so then hthe method will return 'true'
      * and set up a contents of the structure. Otherwise it will return
      * 'false'.
      *
      * EXCEPTIONS:
      *
      *   "WrongParams"   : to report wrong parameters (non-existing experiment, etc.)
      *   "DatabaseError" : to report database related problems
      *
      * @see class WrongParams
      * @see class DatabaseError
      */
     virtual bool getParamInfo (ParamInfo& info,
                                const std::string& instrument,
                                const std::string& experiment,
                                const std::string& parameter) throw (WrongParams,
                                                                     DatabaseError) = 0 ;

    // --------------------------------------------------------------------
    // ------------- Database contents modification methods ---------------
    // --------------------------------------------------------------------

    /**
      * Create a placeholder for a new run.
      *
      * The method would create a new run entry in the database. The run
      * will be assigned the specified number and it will have the range.
      * The range should fit into the experiment's range.
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
      * The connection will be open using the specified ODBC connection string.
      * In case of a success, a valid pointer is returned. The object's ownership
      * is also retured by the method.
      *
      * EXCEPTIONS:
      *
      *   "DatabaseError" : to report database related problems
      *
      * @see class DatabaseError
      */
    static Connection* open (const std::string& odbc_conn_str) throw (DatabaseError) ;

    // Selectors (const)

    // Modifiers

private:

    // Data members

    static Connection* s_conn ; // cached connection (if any)
} ;

} // namespace SciMD

#endif // SCIMD_CONNECTION_H
