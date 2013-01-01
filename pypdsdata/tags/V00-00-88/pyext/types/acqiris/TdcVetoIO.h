#ifndef PYPDSDATA_ACQIRIS_TDCVETOIO_H
#define PYPDSDATA_ACQIRIS_TDCVETOIO_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcVetoIO.
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

class TdcVetoIO : public PdsDataType<TdcVetoIO,Pds::Acqiris::TdcVetoIO> {
public:

  typedef PdsDataType<TdcVetoIO,Pds::Acqiris::TdcVetoIO> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Acqiris
} // namespace pypdsdata

#endif // PYPDSDATA_ACQIRIS_TDCVETOIO_H
