#ifndef PYPDSDATA_CSPAD_CSPADDIGITALPOTSCFG_H
#define PYPDSDATA_CSPAD_CSPADDIGITALPOTSCFG_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadDigitalPotsCfg.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../PdsDataType.h"

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

class CsPadDigitalPotsCfg : public PdsDataType<CsPadDigitalPotsCfg, Pds::CsPad::CsPadDigitalPotsCfg> {
public:

  typedef PdsDataType<CsPadDigitalPotsCfg, Pds::CsPad::CsPadDigitalPotsCfg> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace CsPad
} // namespace pypdsdata

#endif // PYPDSDATA_CSPAD_CSPADDIGITALPOTSCFG_H
