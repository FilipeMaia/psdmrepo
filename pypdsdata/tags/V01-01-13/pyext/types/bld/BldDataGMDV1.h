#ifndef PYPDSDATA_BLD_BLDDATAGMDV1_H
#define PYPDSDATA_BLD_BLDDATAGMDV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataGMDV1.
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
#include "pdsdata/psddl/bld.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Bld {

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

class BldDataGMDV1 : public PdsDataType<BldDataGMDV1,Pds::Bld::BldDataGMDV1> {
public:

  typedef PdsDataType<BldDataGMDV1,Pds::Bld::BldDataGMDV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Bld
} // namespace pypdsdata

#endif // PYPDSDATA_BLD_BLDDATAGMDV1_H
