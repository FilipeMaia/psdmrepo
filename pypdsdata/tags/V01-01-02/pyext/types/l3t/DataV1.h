#ifndef PYPDSDATA_L3T_DATAV1_H
#define PYPDSDATA_L3T_DATAV1_H

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
#include "../PdsDataTypeEmbedded.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/psddl/l3t.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace L3T {

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

class DataV1 : public PdsDataTypeEmbedded<DataV1, Pds::L3T::DataV1> {
public:

  typedef PdsDataTypeEmbedded<DataV1, Pds::L3T::DataV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace L3T
} // namespace pypdsdata

namespace Pds {
namespace L3T {
inline PyObject* toPython(const Pds::L3T::DataV1& v) { return pypdsdata::L3T::DataV1::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_L3T_DATAV1_H
