#ifndef PYPDSDATA_OCEANOPTICS_DATAV2_H
#define PYPDSDATA_OCEANOPTICS_DATAV2_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class DataV2.
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
#include "pdsdata/psddl/oceanoptics.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace OceanOptics {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class DataV2 : public PdsDataType<DataV2, Pds::OceanOptics::DataV2> {
public:

  typedef PdsDataType<DataV2, Pds::OceanOptics::DataV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace OceanOptics
} // namespace pypdsdata

#endif // PYPDSDATA_OCEANOPTICS_DATAV2_H
