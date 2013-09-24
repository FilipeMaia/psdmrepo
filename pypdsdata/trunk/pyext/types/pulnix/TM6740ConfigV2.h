#ifndef PYPDSDATA_PULNIX_TM6740CONFIGV2_H
#define PYPDSDATA_PULNIX_TM6740CONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Pulnix_TM6740ConfigV2.
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
#include "pdsdata/psddl/pulnix.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Pulnix {

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

class TM6740ConfigV2 : public PdsDataType<TM6740ConfigV2,Pds::Pulnix::TM6740ConfigV2> {
public:

  typedef PdsDataType<TM6740ConfigV2,Pds::Pulnix::TM6740ConfigV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Pulnix
} // namespace pypdsdata

#endif // PYPDSDATA_PULNIX_TM6740CONFIGV2_H
