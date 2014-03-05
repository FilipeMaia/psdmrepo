#ifndef PYPDSDATA_PARTITION_SOURCE_H
#define PYPDSDATA_PARTITION_SOURCE_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class Source.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/psddl/partition.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Partition {

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

class Source : public PdsDataTypeEmbedded<Source, Pds::Partition::Source> {
public:

  typedef PdsDataTypeEmbedded<Source, Pds::Partition::Source> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Partition
} // namespace pypdsdata

namespace Pds {
namespace Partition {
inline PyObject* toPython(const Pds::Partition::Source& v) { return pypdsdata::Partition::Source::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_PARTITION_SOURCE_H
