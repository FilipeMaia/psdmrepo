#ifndef PYPDSDATA_PRINCETON_INFOV1_H
#define PYPDSDATA_PRINCETON_INFOV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class InfoV1.
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
#include "pdsdata/psddl/princeton.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace Princeton {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class InfoV1 : public PdsDataTypeEmbedded<InfoV1,Pds::Princeton::InfoV1> {
public:

  typedef PdsDataTypeEmbedded<InfoV1,Pds::Princeton::InfoV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Dump object info to a stream
  void print(std::ostream& out) const;

};

} // namespace Princeton
} // namespace pypdsdata

namespace Pds {
namespace Princeton {
inline PyObject* toPython(const Pds::Princeton::InfoV1& v) { return pypdsdata::Princeton::InfoV1::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_PRINCETON_INFOV1_H
