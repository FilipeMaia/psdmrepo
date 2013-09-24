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
#include "pdsdata/psddl/fccd.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace FCCD {

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

class FccdConfigV2 : public PdsDataType<FccdConfigV2,Pds::FCCD::FccdConfigV2> {
public:

  typedef PdsDataType<FccdConfigV2,Pds::FCCD::FccdConfigV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace FCCD
} // namespace pypdsdata

#endif // PYPDSDATA_FCCD_FCCDCONFIGV2_H
