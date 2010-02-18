#ifndef PYPDSDATA_CONTROLDATA_CONFIGV1_H
#define PYPDSDATA_CONTROLDATA_CONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ControlData_ConfigV1.
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
#include "pdsdata/control/ConfigV1.hh"

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

class ConfigV1 : public PdsDataType<ConfigV1,Pds::ControlData::ConfigV1> {
public:

  typedef PdsDataType<ConfigV1,Pds::ControlData::ConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace ControlData
} // namespace pypdsdata

#endif // PYPDSDATA_CONTROLDATA_CONFIGV1_H
