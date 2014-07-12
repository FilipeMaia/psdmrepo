#ifndef PYPDSDATA_BLD_BLDDATAEBEAMV6_H
#define PYPDSDATA_BLD_BLDDATAEBEAMV6_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: BldDataEBeamV6.h 7826 2014-03-10 22:27:38Z davidsch@SLAC.STANFORD.EDU $
//
// Description:
//	Class BldDataEBeamV6.
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
 *  @version $Id: BldDataEBeamV6.h 7826 2014-03-10 22:27:38Z davidsch@SLAC.STANFORD.EDU $
 *
 *  @author Andrei Salnikov
 */

class BldDataEBeamV6 : public PdsDataType<BldDataEBeamV6,Pds::Bld::BldDataEBeamV6> {
public:

  typedef PdsDataType<BldDataEBeamV6,Pds::Bld::BldDataEBeamV6> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Bld
} // namespace pypdsdata

#endif // PYPDSDATA_BLD_BLDDATAEBEAMV6_H
