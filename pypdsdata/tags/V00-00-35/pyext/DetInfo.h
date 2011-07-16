#ifndef PYPDSDATA_DETINFO_H
#define PYPDSDATA_DETINFO_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DetInfo.
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
#include "pdsdata/xtc/DetInfo.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

/**
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class DetInfo : public PdsDataTypeEmbedded<DetInfo,Pds::DetInfo> {
public:

  typedef PdsDataTypeEmbedded<DetInfo,Pds::DetInfo> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

#endif // PYPDSDATA_DETINFO_H
