#ifndef PYPDSDATA_ACQIRIS_TDCDATAV1MARKER_H
#define PYPDSDATA_ACQIRIS_TDCDATAV1MARKER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcDataV1Marker.
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
#include "pdsdata/psddl/acqiris.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Acqiris {

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

class TdcDataV1Marker : public PdsDataTypeEmbedded<TdcDataV1Marker,class Pds::Acqiris::TdcDataV1Marker> {
public:

  typedef PdsDataTypeEmbedded<TdcDataV1Marker,class Pds::Acqiris::TdcDataV1Marker> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Acqiris
} // namespace pypdsdata

namespace Pds {
namespace Acqiris {
inline PyObject* toPython(const Pds::Acqiris::TdcDataV1Marker& v) { return pypdsdata::Acqiris::TdcDataV1Marker::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_ACQIRIS_TDCDATAV1MARKER_H
