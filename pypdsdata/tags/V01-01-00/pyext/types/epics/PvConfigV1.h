#ifndef PYPDSDATA_EPICS_PVCONFIGV1_H
#define PYPDSDATA_EPICS_PVCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class PvConfigV1.
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
#include "pdsdata/psddl/epics.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Epics {

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

class PvConfigV1 : public PdsDataTypeEmbedded<PvConfigV1, Pds::Epics::PvConfigV1> {
public:

  typedef PdsDataTypeEmbedded<PvConfigV1, Pds::Epics::PvConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Epics
} // namespace pypdsdata

namespace Pds {
namespace Epics {
inline PyObject* toPython(const Pds::Epics::PvConfigV1& v) { return pypdsdata::Epics::PvConfigV1::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_EPICS_PVCONFIGV1_H
