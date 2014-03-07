#ifndef PYPDSDATA_CONTROLDATA_PVCONTROL_H
#define PYPDSDATA_CONTROLDATA_PVCONTROL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_PVControl.
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
#include "pdsdata/psddl/control.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace ControlData {

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

class PVControl : public PdsDataTypeEmbedded<PVControl,Pds::ControlData::PVControl> {
public:

  typedef PdsDataTypeEmbedded<PVControl,Pds::ControlData::PVControl> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace ControlData
} // namespace pypdsdata

namespace Pds {
namespace ControlData {
inline PyObject* toPython(const Pds::ControlData::PVControl& v) { return pypdsdata::ControlData::PVControl::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_CONTROLDATA_PVCONTROL_H
