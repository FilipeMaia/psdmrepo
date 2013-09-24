#ifndef PYPDSDATA_EPICSTIMESTAMP_H
#define PYPDSDATA_EPICSTIMESTAMP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class epicsTimeStamp.
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
#include "pdsdata/psddl/epics.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Epics {

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

class epicsTimeStamp : public PdsDataTypeEmbedded<epicsTimeStamp,Pds::Epics::epicsTimeStamp> {
public:

  typedef PdsDataTypeEmbedded<epicsTimeStamp,Pds::Epics::epicsTimeStamp> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Epics
} // namespace pypdsdata

namespace Pds {
namespace Epics {
inline PyObject* toPython(const Pds::Epics::epicsTimeStamp& v) { return pypdsdata::Epics::epicsTimeStamp::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_EPICSTIMESTAMP_H
