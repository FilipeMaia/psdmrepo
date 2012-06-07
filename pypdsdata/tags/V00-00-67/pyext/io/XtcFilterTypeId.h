#ifndef PYPDSDATA_XTCFILTERTYPEID_H
#define PYPDSDATA_XTCFILTERTYPEID_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFilterTypeId.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../types/PdsDataTypeEmbedded.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "XtcInput/XtcFilterTypeId.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  @brief Python wrapper for C++ type XtcInput::XtcFilterTypeId
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class XtcFilterTypeId : public PdsDataTypeEmbedded<XtcFilterTypeId, XtcInput::XtcFilterTypeId> {
public:

  typedef PdsDataTypeEmbedded<XtcFilterTypeId, XtcInput::XtcFilterTypeId> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

#endif // PYPDSDATA_XTCFILTERTYPEID_H
