#ifndef ROOTHIST_EXCEPTIONS_H
#define ROOTHIST_EXCEPTIONS_H

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

//----------------------
// Base Class Headers --
//----------------------
#include "ErrSvc/Issue.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace RootHist {

/**
 *  @ingroup RootHist
 *  
 *  @brief Base class for exceptions for RootHist package.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Exception : public ErrSvc::Issue {
public:

  /// Constructor takes the reason for an exception
  Exception(const ErrSvc::Context& ctx, const std::string& what) ;

};

/// Exception thrown for failures during file opening
class ExceptionFileOpen : public Exception {
public:

  /// Constructor takes the name of the file
  ExceptionFileOpen(const ErrSvc::Context& ctx, const std::string& file) ;

};

} // namespace RootHist

#endif // ROOTHIST_EXCEPTIONS_H
