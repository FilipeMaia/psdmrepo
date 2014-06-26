#ifndef PSCALIB_EXCEPTIONS_H
#define PSCALIB_EXCEPTIONS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// $Revision$
//------------------------------------------------------------------------

//----------------------
// Base Class Headers --
//----------------------
#include "ErrSvc/Issue.h"

//----------------------------

namespace PSCalib {

/// @addtogroup PSCalib

/**
 *  @ingroup PSCalib
 *
 *  @brief Base class for exception classes for PSCalib package.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Exception : public ErrSvc::Issue {
public:

  /// Constructor takes the reason for an exception
  Exception ( const ErrSvc::Context& ctx, const std::string& what ) ;

};

/// Exception thrown when Source address is not DetInfo
class NotDetInfoError : public Exception {
public:

  NotDetInfoError ( const ErrSvc::Context& ctx ) ;

};

} // namespace PSCalib

#endif // PSCALIB_EXCEPTIONS_H
