#ifndef PYPDSDATA_EVRDATA_SRCEVENTCODE_H
#define PYPDSDATA_EVRDATA_SRCEVENTCODE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_SrcEventCode.
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

class SrcEventCode : public PdsDataTypeEmbedded<SrcEventCode,Pds::EvrData::SrcEventCode> {
public:

  typedef PdsDataTypeEmbedded<SrcEventCode,Pds::EvrData::SrcEventCode> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;
};

} // namespace EvrData
} // namespace pypdsdata

namespace Pds {
namespace EvrData {
inline PyObject* toPython(const Pds::EvrData::SrcEventCode& v) { return pypdsdata::EvrData::SrcEventCode::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_EVRDATA_SRCEVENTCODE_H
