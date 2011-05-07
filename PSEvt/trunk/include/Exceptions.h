#ifndef PSEVT_EXCEPTIONS_H
#define PSEVT_EXCEPTIONS_H

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
#include "pdsdata/xtc/Src.hh"
#include "PSEvt/EventKey.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSEvt {

/**
 *  @brief Base class for exception classes for PSEvt package.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version \$Id$
 *
 *  @author Andrei Salnikov
 */

class Exception : public ErrSvc::Issue {
public:

  /// Constructor takes the reason for an exception
  Exception ( const ErrSvc::Context& ctx, const std::string& what ) ;

};

/// Exception thrown when trying to store multiple objects with the same key
class ExceptionDuplicateKey : public Exception {
public:

  ExceptionDuplicateKey ( const ErrSvc::Context& ctx, const EventKey& key ) ;

};

/// Exception thrown when Source format string is not recognized
class ExceptionSourceFormat : public Exception {
public:

  ExceptionSourceFormat ( const ErrSvc::Context& ctx, const std::string& format ) ;

};

} // namespace PSEvt

#endif // PSEVT_EXCEPTIONS_H
