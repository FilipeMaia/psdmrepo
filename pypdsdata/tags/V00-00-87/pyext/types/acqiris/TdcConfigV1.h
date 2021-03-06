#ifndef PYPDSDATA_ACQIRIS_TDCCONFIGV1_H
#define PYPDSDATA_ACQIRIS_TDCCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcConfigV1.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/acqiris/TdcConfigV1.hh"

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

class TdcConfigV1 : public PdsDataType<TdcConfigV1,Pds::Acqiris::TdcConfigV1> {
public:

  typedef PdsDataType<TdcConfigV1,Pds::Acqiris::TdcConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );
  
  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Acqiris
} // namespace pypdsdata

#endif // PYPDSDATA_ACQIRIS_TDCCONFIGV1_H
