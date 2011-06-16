#ifndef PYPDSDATA_ACQIRIS_TDCDATAV1_H
#define PYPDSDATA_ACQIRIS_TDCDATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcDataV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../PdsDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/acqiris/TdcDataV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
namespace pypdsdata {
  class EnumType;
}

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Acqiris {

/**
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class TdcDataV1 : public PdsDataType<TdcDataV1,Pds::Acqiris::TdcDataV1> {
public:

  typedef PdsDataType<TdcDataV1,Pds::Acqiris::TdcDataV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  /// Returns enum type object for Source enum
  static const EnumType& sourceEnum();
  
  // dump to a stream
  void print(std::ostream& out) const;
  
};

} // namespace Acqiris
} // namespace pypdsdata

#endif // PYPDSDATA_ACQIRIS_TDCDATAV1_H
