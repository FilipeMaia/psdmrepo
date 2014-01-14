#ifndef PYPDSDATA_PRINCETON_CONFIGV3_H
#define PYPDSDATA_PRINCETON_CONFIGV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV3.
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
#include "pdsdata/psddl/princeton.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace Princeton {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class ConfigV3 : public PdsDataType<ConfigV3,Pds::Princeton::ConfigV3> {
public:

  typedef PdsDataType<ConfigV3,Pds::Princeton::ConfigV3> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Dump object info to a stream
  void print(std::ostream& out) const;

};

} // namespace Princeton
} // namespace pypdsdata

#endif // PYPDSDATA_PRINCETON_CONFIGV3_H
