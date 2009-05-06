#ifndef SCIMD_VALUETYPEMISMATCH_H
#define SCIMD_VALUETYPEMISMATCH_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
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

namespace SciMD {

/**
 * Exception class to report wrong type usage in class or method templates.
 *
 * USAGE:
 *
 * This software was developed for the LUSI project.  If you use all or 
 * part of it, please give an appropriate acknowledgement.
 *
 * @version $Id: template!C++!h 4 2008-10-08 19:27:36Z salnikov $
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

} // namespace SciMD

#endif // SCIMD_VALUETYPEMISMATCH_H
