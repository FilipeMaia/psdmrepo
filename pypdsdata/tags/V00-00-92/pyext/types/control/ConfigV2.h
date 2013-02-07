#ifndef PYPDSDATA_CONTROLDATA_CONFIGV2_H
#define PYPDSDATA_CONTROLDATA_CONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_ConfigV2.
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
#include "pdsdata/control/ConfigV2.hh"

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

class ConfigV2 : public PdsDataType<ConfigV2,Pds::ControlData::ConfigV2> {
public:

  typedef PdsDataType<ConfigV2,Pds::ControlData::ConfigV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace ControlData
} // namespace pypdsdata

#endif // PYPDSDATA_CONTROLDATA_CONFIGV2_H
