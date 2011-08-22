#ifndef PYPDSDATA_FCCD_FCCDCONFIGV1_H
#define PYPDSDATA_FCCD_FCCDCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FccdConfigV1.
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
#include "pdsdata/fccd/FccdConfigV1.hh"

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

class FccdConfigV1 : public PdsDataType<FccdConfigV1,Pds::FCCD::FccdConfigV1> {
public:

  typedef PdsDataType<FccdConfigV1,Pds::FCCD::FccdConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace FCCD
} // namespace pypdsdata

#endif // PYPDSDATA_FCCD_FCCDCONFIGV1_H
