#ifndef PYPDSDATA_OPAL1K_CONFIGV1_H
#define PYPDSDATA_OPAL1K_CONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1.
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
#include "pdsdata/psddl/opal1k.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Opal1k {

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

class ConfigV1 : public PdsDataType<ConfigV1,Pds::Opal1k::ConfigV1> {
public:

  typedef PdsDataType<ConfigV1,Pds::Opal1k::ConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Opal1k
} // namespace pypdsdata

#endif // PYPDSDATA_OPAL1K_CONFIGV1_H
