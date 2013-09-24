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

namespace O2OTranslator {

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

template <typename T>
class DataTypeCvt : public DataTypeCvtI {
public:

  // Destructor
  virtual ~DataTypeCvt () {}

  /// main method of this class
  virtual void convert(const void* data,
                       size_t size,
                       const Pds::TypeId& typeId,
                       const O2OXtcSrc& src,
                       const H5DataTypes::XtcClockTimeStamp& time,
                       Pds::Damage damage)
  {
    if (damage.value() == 0 or
        damage.value() == (1 << Pds::Damage::OutOfOrder) or
        (typeId.id() == Pds::TypeId::Id_EBeam and damage.bits() == (1 << Pds::Damage::UserDefined))) {
      // All non-damaged data, out-of-order data, or BLD Ebeam data with only user
      // damage are passed to conversion method
      const T& typedData = *static_cast<const T*>( data ) ;
      typedConvert(typedData, size, typeId, src, time, damage);
    } else {
      // for damaged data we don't want to look at the data so call special
      // method to fill the gaps
      missingConvert(typeId, src, time, damage);
    }
  }

protected:

  // Default constructor
  DataTypeCvt () : DataTypeCvtI() {}

private:

  // typed conversion method
  virtual void typedConvert(const T& data,
                            size_t size,
                            const Pds::TypeId& typeId,
                            const O2OXtcSrc& src,
                            const H5DataTypes::XtcClockTimeStamp& time,
                            Pds::Damage damage) = 0 ;

  // method called to fill void spaces for missing data
  virtual void missingConvert(const Pds::TypeId& typeId,
                              const O2OXtcSrc& src,
                              const H5DataTypes::XtcClockTimeStamp& time,
                              Pds::Damage damage) = 0 ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_DATATYPECVT_H
