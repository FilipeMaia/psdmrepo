#ifndef PSDDL_HDF2PSANA_EXCEPTIONS_H
#define PSDDL_HDF2PSANA_EXCEPTIONS_H

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

namespace psddl_hdf2psana {

/// @addtogroup psddl_hdf2psana

/**
 *  @ingroup psddl_hdf2psana
 *
 *  @brief Base class for exception classes for psddl_hdf2psana package.
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

/// Exception thrown when group name cannot be converted to source address
class ExceptionGroupSourceName : public Exception {
public:

  ExceptionGroupSourceName ( const ErrSvc::Context& ctx, const std::string& group ) ;

};

/// Exception thrown when group name cannot be converted to TypeId
class ExceptionGroupTypeIdName : public Exception {
public:

  ExceptionGroupTypeIdName ( const ErrSvc::Context& ctx, const std::string& group ) ;

};

/// Exception thrown when data rank is not as expected
class ExceptionDataRank : public Exception {
public:

  ExceptionDataRank( const ErrSvc::Context& ctx, int rank, int expectedRank ) ;

};

/// Exception thrown when schema version number is not known
class ExceptionSchemaVersion : public Exception {
public:

  ExceptionSchemaVersion( const ErrSvc::Context& ctx, const std::string& type, int version ) ;

};

/// Exception thrown when call is made to unimplemented method
class ExceptionNotImplemented : public Exception {
public:

  ExceptionNotImplemented( const ErrSvc::Context& ctx, const std::string& msg ) ;

};


} // namespace psddl_hdf2psana

#endif // PSDDL_HDF2PSANA_EXCEPTIONS_H
