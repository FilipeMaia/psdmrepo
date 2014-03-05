#ifndef PYPDSDATA_TIMEPIX_CONFIGV3_H
#define PYPDSDATA_TIMEPIX_CONFIGV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Timepix_ConfigV3.
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
#include "pdsdata/psddl/timepix.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Timepix {

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

class ConfigV3 : public PdsDataType<ConfigV3,Pds::Timepix::ConfigV3> {
public:

  typedef PdsDataType<ConfigV3,Pds::Timepix::ConfigV3> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Timepix
} // namespace pypdsdata

#endif // PYPDSDATA_TIMEPIX_CONFIGV3_H
