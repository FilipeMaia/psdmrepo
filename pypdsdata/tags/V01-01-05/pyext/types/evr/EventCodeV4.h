#ifndef PYPDSDATA_EVRDATA_EVENTCODEV4_H
#define PYPDSDATA_EVRDATA_EVENTCODEV4_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_EventCodeV4.
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

class EventCodeV4 : public PdsDataTypeEmbedded<EventCodeV4,Pds::EvrData::EventCodeV4> {
public:

  typedef PdsDataTypeEmbedded<EventCodeV4,Pds::EvrData::EventCodeV4> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace EvrData
} // namespace pypdsdata

namespace Pds {
namespace EvrData {
inline PyObject* toPython(const Pds::EvrData::EventCodeV4& v) { return pypdsdata::EvrData::EventCodeV4::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_EVRDATA_EVENTCODEV4_H
