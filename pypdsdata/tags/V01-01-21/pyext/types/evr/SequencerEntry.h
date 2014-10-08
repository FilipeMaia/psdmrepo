#ifndef PYPDSDATA_EVRDATA_SEQUENCERENTRY_H
#define PYPDSDATA_EVRDATA_SEQUENCERENTRY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_SequencerEntry.
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
#include "pdsdata/psddl/evr.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace EvrData {

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

class SequencerEntry : public PdsDataTypeEmbedded<SequencerEntry,Pds::EvrData::SequencerEntry> {
public:

  typedef PdsDataTypeEmbedded<SequencerEntry,Pds::EvrData::SequencerEntry> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace EvrData
} // namespace pypdsdata

namespace Pds {
namespace EvrData {
inline PyObject* toPython(const Pds::EvrData::SequencerEntry& v) { return pypdsdata::EvrData::SequencerEntry::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_EVRDATA_SEQUENCERENTRY_H
