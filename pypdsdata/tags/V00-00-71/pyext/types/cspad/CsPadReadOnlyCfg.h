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
#include "../PdsDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/cspad/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace pypdsdata {
namespace CsPad {

/**
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPadReadOnlyCfg : public PdsDataType<CsPadReadOnlyCfg, Pds::CsPad::CsPadReadOnlyCfg> {
public:

  typedef PdsDataType<CsPadReadOnlyCfg, Pds::CsPad::CsPadReadOnlyCfg> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace CsPad
} // namespace pypdsdata

#endif // PYPDSDATA_CSPAD_CSPADREADONLYCFG_H
