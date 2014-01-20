#ifndef PYPDSDATA_ACQIRIS_TDCCHANNEL_H
#define PYPDSDATA_ACQIRIS_TDCCHANNEL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcChannel.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../PdsDataTypeEmbedded.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/psddl/acqiris.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Acqiris {

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

class TdcChannel : public PdsDataTypeEmbedded<TdcChannel,Pds::Acqiris::TdcChannel> {
public:

  typedef PdsDataTypeEmbedded<TdcChannel,Pds::Acqiris::TdcChannel> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Acqiris
} // namespace pypdsdata

namespace Pds {
namespace Acqiris {
inline PyObject* toPython(const Pds::Acqiris::TdcChannel& v) { return pypdsdata::Acqiris::TdcChannel::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_ACQIRIS_TDCCHANNEL_H
