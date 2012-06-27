#ifndef PYPDSDATA_OCEANOPTICS_DATAV1_H
#define PYPDSDATA_OCEANOPTICS_DATAV1_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class DataV1.
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
#include "pdsdata/oceanoptics/DataV1.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace OceanOptics {

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

class DataV1 : public PdsDataType<DataV1, Pds::OceanOptics::DataV1> {
public:

  typedef PdsDataType<DataV1, Pds::OceanOptics::DataV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace OceanOptics
} // namespace pypdsdata

#endif // PYPDSDATA_OCEANOPTICS_DATAV1_H
