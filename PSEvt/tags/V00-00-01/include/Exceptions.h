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
#include <stdexcept>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/DetInfo.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PsEvt {

/**
 *  @brief Exception classes for PsEvt package.
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

class Exception : public std::runtime_error {
public:

  /// Constructor takes the reason for an exception
  Exception ( const std::string& what ) ;

};

/// Exception thrown when rtying to store multiple objects with the same key
class ExceptionDuplicateKey : public Exception {
public:

  ExceptionDuplicateKey ( const std::type_info* typeinfo, 
                          const Pds::DetInfo& detInfo, 
                          const std::string& key ) ;

};

} // namespace PsEvt

#endif // PSEVT_EXCEPTIONS_H
