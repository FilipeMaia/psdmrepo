#ifndef PYPDSDATA_CSPAD_CSPADREADONLYCFG_H
#define PYPDSDATA_CSPAD_CSPADREADONLYCFG_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadReadOnlyCfg.
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
#include "pdsdata/psddl/cspad.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace pypdsdata {
namespace CsPad {

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

class CsPadReadOnlyCfg : public PdsDataTypeEmbedded<CsPadReadOnlyCfg, Pds::CsPad::CsPadReadOnlyCfg> {
public:

  typedef PdsDataTypeEmbedded<CsPadReadOnlyCfg, Pds::CsPad::CsPadReadOnlyCfg> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace CsPad
} // namespace pypdsdata

namespace Pds {
namespace CsPad {
inline PyObject* toPython(const Pds::CsPad::CsPadReadOnlyCfg& v) { return pypdsdata::CsPad::CsPadReadOnlyCfg::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_CSPAD_CSPADREADONLYCFG_H
