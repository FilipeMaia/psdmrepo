#ifndef PYPDSDATA_EVRDATA_PULSECONFIGV3_H
#define PYPDSDATA_EVRDATA_PULSECONFIGV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_PulseConfigV3.
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
#include "pdsdata/psddl/evr.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace EvrData {

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

class PulseConfigV3 : public PdsDataTypeEmbedded<PulseConfigV3,Pds::EvrData::PulseConfigV3> {
public:

  typedef PdsDataTypeEmbedded<PulseConfigV3,Pds::EvrData::PulseConfigV3> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace EvrData
} // namespace pypdsdata

namespace Pds {
namespace EvrData {
inline PyObject* toPython(const Pds::EvrData::PulseConfigV3& v) { return pypdsdata::EvrData::PulseConfigV3::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_EVRDATA_PULSECONFIGV3_H
