#ifndef PYPDSDATA_FCCD_FCCDCONFIGV2_H
#define PYPDSDATA_FCCD_FCCDCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FccdConfigV2.
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
#include "pdsdata/fccd/FccdConfigV2.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace FCCD {

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

class FccdConfigV2 : public PdsDataType<FccdConfigV2,Pds::FCCD::FccdConfigV2> {
public:

  typedef PdsDataType<FccdConfigV2,Pds::FCCD::FccdConfigV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace FCCD
} // namespace pypdsdata

#endif // PYPDSDATA_FCCD_FCCDCONFIGV2_H
