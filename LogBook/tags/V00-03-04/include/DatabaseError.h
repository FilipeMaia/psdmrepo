#ifndef LOGBOOK_DATABASEERROR_H
#define LOGBOOK_DATABASEERROR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: $
//
// Description:
//	Class DatabaseError.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include <string>

//----------------------
// Base Class Headers --
//----------------------

#include <exception>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace LogBook {

/**
 * Exception class for handling database related problems.
 *
 * USAGE:
 *
 * This software was developed for the LCLS project.  If you use all or 
 * part of it, please give an appropriate acknowledgement.
 *
 * @version $Id: $
 *
 * @author Igor Gaponenko
 */
class DatabaseError : public std::exception {

public:

    /**
      * Normal constructor.
      *
      * Create an object and explain a reason of the exception.
      */
    explicit DatabaseError (const std::string& reason) ;

    /**
      * Copy constructor.
      */
    DatabaseError ( const DatabaseError& ) ;

    /**
      * Assignment operator.
      */
    DatabaseError& operator = ( const DatabaseError& ) ;

    // Destructor

    virtual ~DatabaseError () throw() ;

    /**
      * Return a reason of the exception.
      *
      * The method overrides the corresponding method of the derived
      * class.
      *
      * @see method std::exception::what()
      */
    virtual const char* what () const throw() ;

private:

    // Default constructor is disabled

    DatabaseError () ;

private:

    // Data members

    std::string m_reason;  // what caused the exception
};

} // namespace LogBook

#endif // LOGBOOK_DATABASEERROR_H
