#ifndef PYPDSDATA_PROCINFO_H
#define PYPDSDATA_PROCINFO_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProcInfo.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "types/PdsDataTypeEmbedded.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/xtc/ProcInfo.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {

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

class ProcInfo : public PdsDataTypeEmbedded<ProcInfo,Pds::ProcInfo> {
public:

  typedef PdsDataTypeEmbedded<ProcInfo,Pds::ProcInfo> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

namespace Pds {
inline PyObject* toPython(const Pds::ProcInfo& v) { return pypdsdata::ProcInfo::PyObject_FromPds(v); }
}

#endif // PYPDSDATA_PROCINFO_H
