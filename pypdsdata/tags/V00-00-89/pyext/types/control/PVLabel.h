#ifndef PYPDSDATA_CONTROLDATA_PVLABEL_H
#define PYPDSDATA_CONTROLDATA_PVLABEL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_PVLabel.
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
#include "pdsdata/control/PVLabel.hh"

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

class PVLabel : public PdsDataType<PVLabel,Pds::ControlData::PVLabel> {
public:

  typedef PdsDataType<PVLabel,Pds::ControlData::PVLabel> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace ControlData
} // namespace pypdsdata

#endif // PYPDSDATA_CONTROLDATA_PVLABEL_H
