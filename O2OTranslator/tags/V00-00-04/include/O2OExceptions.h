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

//----------------------
// Base Class Headers --
//----------------------
#include <stdexcept>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Exception classes
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace O2OTranslator {

class O2OException : public std::runtime_error {
public:

  // Constructor
  O2OException ( const std::string& className, const std::string& what ) ;

  virtual ~O2OException () throw() ;

};


class O2OFileOpenException : public O2OException {
public:

  O2OFileOpenException( const std::string& fileName )
    : O2OException( "O2OFileOpenException", "failed to open file "+fileName ) {}

  virtual ~O2OFileOpenException() throw() ;
};

class O2ONexusException : public O2OException {
public:

  O2ONexusException( const std::string& function )
    : O2OException( "O2ONexusException", "Nexus error in call to function "+function ) {}

  virtual ~O2ONexusException() throw() ;
};


} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OEXCEPTIONS_H
