#ifndef PYPDSDATA_USDUSB_DATAV1_H
#define PYPDSDATA_USDUSB_DATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Pulnix_DataV1.
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
#include "pdsdata/usdusb/DataV1.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace UsdUsb {

/**
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class DataV1 : public PdsDataType<DataV1, Pds::UsdUsb::DataV1> {
public:

  typedef PdsDataType<DataV1, Pds::UsdUsb::DataV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace UsdUsb
} // namespace pypdsdata

#endif // PYPDSDATA_USDUSB_DATAV1_H
