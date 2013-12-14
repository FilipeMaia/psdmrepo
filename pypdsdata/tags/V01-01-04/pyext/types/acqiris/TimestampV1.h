#ifndef PYPDSDATA_ACQIRIS_TIMESTAMPV1_H
#define PYPDSDATA_ACQIRIS_TIMESTAMPV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TimestampV1.
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

class TimestampV1 : public PdsDataTypeEmbedded<TimestampV1,Pds::Acqiris::TimestampV1> {
public:

  typedef PdsDataTypeEmbedded<TimestampV1,Pds::Acqiris::TimestampV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Acqiris
} // namespace pypdsdata

namespace Pds {
namespace Acqiris {
inline PyObject* toPython(const Pds::Acqiris::TimestampV1& v) { return pypdsdata::Acqiris::TimestampV1::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_ACQIRIS_TIMESTAMPV1_H
