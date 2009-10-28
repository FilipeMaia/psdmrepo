#ifndef O2OTRANSLATOR_DATATYPECVT_H
#define O2OTRANSLATOR_DATATYPECVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DataTypeCvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvtI.h"

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

template <typename T>
class DataTypeCvt : public DataTypeCvtI {
public:

  // Destructor
  virtual ~DataTypeCvt () {}

  /// main method of this class
  virtual void convert ( const void* data,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src,
                         const H5DataTypes::XtcClockTime& time )
  {
    const T& typedData = *static_cast<const T*>( data ) ;
    typedConvert ( typedData, typeId, src, time ) ;
  }

protected:

  // Default constructor
  DataTypeCvt () : DataTypeCvtI() {}

private:

  // typed conversion method
  virtual void typedConvert ( const T& data,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src,
                              const H5DataTypes::XtcClockTime& time ) = 0 ;

  // Copy constructor and assignment are disabled by default
  DataTypeCvt ( const DataTypeCvt& ) ;
  DataTypeCvt operator = ( const DataTypeCvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_DATATYPECVT_H
