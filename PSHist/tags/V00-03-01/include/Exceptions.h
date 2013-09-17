#ifndef PSHIST_EXCEPTIONS_H
#define PSHIST_EXCEPTIONS_H

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

namespace PSHist {

/**
 *  @ingroup PSHist
 *  
 *  @brief Base class for exceptions for PSHist package.
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
  Exception(const ErrSvc::Context& ctx, const std::string& what) ;

};

/// Exception thrown when histogram or tuple with identical name already exists
class ExceptionDuplicateName : public Exception {
public:

  /// Constructor takes the name of the histogram
  ExceptionDuplicateName(const ErrSvc::Context& ctx, const std::string& name) ;

};

/// Exception thrown when number of bins is 0.
class ExceptionBins : public Exception {
public:

  /// Constructor takes no additional parameters
  ExceptionBins(const ErrSvc::Context& ctx) ;

};

/// Exception thrown when axis low range is same or higher than high range
class ExceptionAxisRange : public Exception {
public:

  /// Constructor takes low and high range
  ExceptionAxisRange(const ErrSvc::Context& ctx, double xlow, double xhigh) ;

};

/// Exception thrown when axis low range is same or higher than high range
class ExceptionAxisEdgeOrder : public Exception {
public:

  /// Constructor takes no additional parameters
  ExceptionAxisEdgeOrder(const ErrSvc::Context& ctx) ;

};

/// Exception thrown when manager files to store histograms
class ExceptionStore : public Exception {
public:

  /// Constructor takes the string which describes the reason
  ExceptionStore(const ErrSvc::Context& ctx, const std::string& reason) ;

};

/// Exception thrown when tuple column with identical name already exists
class ExceptionDuplicateColumn : public Exception {
public:

  /// Constructor takes the name of the tuple and name of the column
  ExceptionDuplicateColumn(const ErrSvc::Context& ctx, const std::string& tupleName, 
      const std::string& columnName) ;

};

} // namespace PSHist

#endif // PSHIST_EXCEPTIONS_H
