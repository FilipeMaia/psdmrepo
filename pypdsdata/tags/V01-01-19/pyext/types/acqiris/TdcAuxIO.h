#ifndef PYPDSDATA_ACQIRIS_TDCAUXIO_H
#define PYPDSDATA_ACQIRIS_TDCAUXIO_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcAuxIO.
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

class TdcAuxIO : public PdsDataTypeEmbedded<TdcAuxIO,Pds::Acqiris::TdcAuxIO> {
public:

  typedef PdsDataTypeEmbedded<TdcAuxIO,Pds::Acqiris::TdcAuxIO> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Acqiris
} // namespace pypdsdata

namespace Pds {
namespace Acqiris {
inline PyObject* toPython(const Pds::Acqiris::TdcAuxIO& v) { return pypdsdata::Acqiris::TdcAuxIO::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_ACQIRIS_TDCAUXIO_H
