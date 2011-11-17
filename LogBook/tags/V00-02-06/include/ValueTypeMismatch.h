#ifndef LOGBOOK_VALUETYPEMISMATCH_H
#define LOGBOOK_VALUETYPEMISMATCH_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: $
//
// Description:
//	Class ValueTypeMismatch.
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
 * Exception class to report wrong type usage in class or method templates.
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
class ValueTypeMismatch : public std::exception {

public:

    /**
      * Normal constructor.
      *
      * Create an object and explain a reason of the exception.
      */
    explicit ValueTypeMismatch (const std::string& reason) ;

    /**
      * Copy constructor.
      */
    ValueTypeMismatch ( const ValueTypeMismatch& ) ;

    /**
      * Assignment operator.
      */
    ValueTypeMismatch& operator = ( const ValueTypeMismatch& ) ;

    // Destructor

    virtual ~ValueTypeMismatch () throw() ;

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

    ValueTypeMismatch () ;

private:

    // Data members

    std::string m_reason;  // what caused the exception
};

} // namespace LogBook

#endif // LOGBOOK_VALUETYPEMISMATCH_H
