#ifndef LUSITIME_EXCEPTIONS_H
#define LUSITIME_EXCEPTIONS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Exceptions.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <stdexcept>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace LusiTime {

/**
 *  Exception classes for LusiTime package
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Exception : public std::runtime_error {
public:

  // Default constructor
  Exception ( const std::string& what ) ;

};

class ExceptionErrno : public Exception {
public:

  // Default constructor
  ExceptionErrno ( const std::string& what ) ;

};

} // namespace LusiTime

#endif // LUSITIME_EXCEPTIONS_H
