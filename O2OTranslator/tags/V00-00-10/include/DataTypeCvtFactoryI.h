#ifndef O2OTRANSLATOR_DATATYPECVTFACTORYI_H
#define O2OTRANSLATOR_DATATYPECVTFACTORYI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataTypeCvtFactoryI.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

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

/**
 *  Interface for a factory class for converters
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace O2OTranslator {

class DataTypeCvtI ;

class DataTypeCvtFactoryI  {
public:

  // Destructor
  virtual ~DataTypeCvtFactoryI () ;

  // Get the converter for given parameter set
  virtual DataTypeCvtI* converter ( const Pds::DetInfo& detInfo ) = 0 ;

protected:

  // Default constructor
  DataTypeCvtFactoryI () {}

  // helper functor for comparing DetInfo objects
  struct CmpDetInfo {
    bool operator()( const Pds::DetInfo& lhs, const Pds::DetInfo& rhs ) const ;
  };

private:

  // Copy constructor and assignment are disabled by default
  DataTypeCvtFactoryI ( const DataTypeCvtFactoryI& ) ;
  DataTypeCvtFactoryI& operator = ( const DataTypeCvtFactoryI& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_DATATYPECVTFACTORYI_H
