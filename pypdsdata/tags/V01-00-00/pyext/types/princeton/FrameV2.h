#ifndef PYPDSDATA_PRINCETON_FRAMEV2_H
#define PYPDSDATA_PRINCETON_FRAMEV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FrameV2.
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

class FrameV2 : public PdsDataType<FrameV2,Pds::Princeton::FrameV2> {
public:

  typedef PdsDataType<FrameV2,Pds::Princeton::FrameV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Dump object info to a stream
  void print(std::ostream& out) const;

};

} // namespace Princeton
} // namespace pypdsdata

#endif // PYPDSDATA_PRINCETON_FRAMEV2_H
