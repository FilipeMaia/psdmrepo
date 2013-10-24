#ifndef PYPDSDATA_EVRDATA_EVENTCODEV5_H
#define PYPDSDATA_EVRDATA_EVENTCODEV5_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_EventCodeV5.
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

class EventCodeV5 : public PdsDataTypeEmbedded<EventCodeV5,Pds::EvrData::EventCodeV5> {
public:

  typedef PdsDataTypeEmbedded<EventCodeV5,Pds::EvrData::EventCodeV5> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace EvrData
} // namespace pypdsdata

namespace Pds {
namespace EvrData {
inline PyObject* toPython(const Pds::EvrData::EventCodeV5& v) { return pypdsdata::EvrData::EventCodeV5::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_EVRDATA_EVENTCODEV5_H
