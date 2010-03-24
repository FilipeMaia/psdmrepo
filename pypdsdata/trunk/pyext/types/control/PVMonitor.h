#ifndef PYPDSDATA_CONTROLDATA_PVMONITOR_H
#define PYPDSDATA_CONTROLDATA_PVMONITOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_PVMonitor.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "types/PdsDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/control/PVMonitor.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace ControlData {

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

class PVMonitor : public PdsDataType<PVMonitor,Pds::ControlData::PVMonitor> {
public:

  typedef PdsDataType<PVMonitor,Pds::ControlData::PVMonitor> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace ControlData
} // namespace pypdsdata

#endif // PYPDSDATA_CONTROLDATA_PVMONITOR_H
