#ifndef SCIMD_WRONGPARAMS_H
#define SCIMD_WRONGPARAMS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class WrongParams.
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
 * Exception class for reporting inappropriate use of parameters to methods.
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
class WrongParams : public std::exception {

public:

    /**
      * Normal constructor.
      *
      * Create an object and explain a reason of the exception.
      */
    explicit WrongParams (const std::string& reason) ;

    /**
      * Copy constructor.
      */
    WrongParams ( const WrongParams& ) ;

    /**
      * Assignment operator.
      */
    WrongParams& operator = ( const WrongParams& ) ;

    // Destructor

    virtual ~WrongParams () throw() ;

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

    WrongParams () ;

private:

    // Data members

    std::string m_reason;  // what caused the exception
};

} // namespace SciMD

#endif // SCIMD_WRONGPARAMS_H
