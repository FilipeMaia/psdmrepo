#ifndef O2OTRANSLATOR_DATATYPECVTI_H
#define O2OTRANSLATOR_DATATYPECVTI_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataTypeCvtI.
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
#include "pdsdata/xtc/TypeId.hh"
#include "H5DataTypes/XtcClockTime.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Abstract base class for data type converters for event-type objects
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

class DataTypeCvtI  {
public:

  // Destructor
  virtual ~DataTypeCvtI () ;

  /// main method of this class
  virtual void convert ( const void* data,
                         const Pds::TypeId& typeId,
                         const Pds::DetInfo& detInfo,
                         const H5DataTypes::XtcClockTime& time ) = 0 ;

protected:

  // Default constructor
  DataTypeCvtI () {}

private:

  // Data members

  // Copy constructor and assignment are disabled by default
  DataTypeCvtI ( const DataTypeCvtI& ) ;
  DataTypeCvtI& operator = ( const DataTypeCvtI& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_DATATYPECVTI_H
