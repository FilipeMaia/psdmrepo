#ifndef PYPDSDATA_PRINCETON_CONFIGV2_H
#define PYPDSDATA_PRINCETON_CONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV2.
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
#include "pdsdata/princeton/ConfigV2.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace Princeton {

/**
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class ConfigV2 : public PdsDataType<ConfigV2,Pds::Princeton::ConfigV2> {
public:

  typedef PdsDataType<ConfigV2,Pds::Princeton::ConfigV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Princeton
} // namespace pypdsdata

#endif // PYPDSDATA_PRINCETON_CONFIGV2_H
