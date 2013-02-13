#ifndef IDATA_EXCEPTIONS_H
#define IDATA_EXCEPTIONS_H

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

namespace IData {

/// @addtogroup IData

/**
 *  @ingroup IData
 *
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
  Exception(const ErrSvc::Context& ctx, const std::string& what);

};

/// Exception thrown when experiment name is not recognized
class ExpNameException : public Exception {
public:

  ExpNameException(const ErrSvc::Context& ctx, const std::string& name)
    : Exception(ctx, "experiment name or number is not recognized: " + name)
  {}

};

/// Exception thrown when run number is not recognized
class RunNumberSpecException : public Exception {
public:

  RunNumberSpecException(const ErrSvc::Context& ctx, const std::string& spec, const std::string& msg)
    : Exception(ctx, "invalid run number specification: '" + spec + "' -- "+ msg)
  {}

};

/// Error generated when dataset directory is not found
class DatasetDirError : public Exception {
public:

  DatasetDirError(const ErrSvc::Context& ctx, const std::string& str)
    : Exception(ctx, "dataset directory is missing: " + str) {}

};

} // namespace IData

#endif // IDATA_EXCEPTIONS_H
