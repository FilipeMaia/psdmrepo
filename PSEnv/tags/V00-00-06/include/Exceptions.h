#ifndef PSENV_EXCEPTIONS_H
#define PSENV_EXCEPTIONS_H

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

namespace PSEnv {

/**
 *  @brief Exception classes for PSEnv package.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Exception : public ErrSvc::Issue {
public:

  /// Constructor takes the reason for an exception
  Exception ( const ErrSvc::Context& ctx, const std::string& what ) ;

};

/// Exception thrown for unknown EPICS PV name
class ExceptionEpicsName : public Exception {
public:

  ExceptionEpicsName ( const ErrSvc::Context& ctx, 
                       const std::string& pvname ) ;

};

/// Exception thrown for conversion errors for EPICS values
class ExceptionEpicsConversion : public Exception {
public:

  ExceptionEpicsConversion ( const ErrSvc::Context& ctx, 
                             const std::string& pvname,
                             const std::type_info& ti,
                             const std::string& what) ;

};

} // namespace PSEnv

#endif // PSENV_EXCEPTIONS_H
