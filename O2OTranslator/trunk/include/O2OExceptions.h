#ifndef O2OTRANSLATOR_O2OEXCEPTIONS_H
#define O2OTRANSLATOR_O2OEXCEPTIONS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OExceptions.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <cerrno>
#include <boost/lexical_cast.hpp>

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

namespace O2OTranslator {

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  Exception classes
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class O2OException : public ErrSvc::Issue {
public:

  // Constructor
  O2OException ( const ErrSvc::Context& ctx, const std::string& className, const std::string& what ) ;

};

class O2OXTCSizeException : public O2OException {
public:

  O2OXTCSizeException( const ErrSvc::Context& ctx, const std::string& type, size_t expectedSize, size_t xtcSize )
    : O2OException( ctx, "O2OXTCSizeException", type + ": XTC size=" +
        boost::lexical_cast<std::string>(xtcSize) +
        ", expected size=" + boost::lexical_cast<std::string>(expectedSize) ) {}

};

/// Generic XTC exception, just give it a message
class O2OXTCGenException : public O2OException {
public:

  O2OXTCGenException( const ErrSvc::Context& ctx, const std::string& msg )
    : O2OException( ctx, "O2OXTCGenException", msg ) {}

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OEXCEPTIONS_H
