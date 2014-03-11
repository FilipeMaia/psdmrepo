#ifndef PYPDSDATA_PNCCD_FRAMESV1_H
#define PYPDSDATA_PNCCD_FRAMESV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FramesV1.
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
#include "pdsdata/psddl/pnccd.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace PNCCD {

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

class FramesV1 : public PdsDataType<FramesV1,Pds::PNCCD::FramesV1> {
public:

  typedef PdsDataType<FramesV1,Pds::PNCCD::FramesV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace PNCCD
} // namespace pypdsdata

#endif // PYPDSDATA_PNCCD_FRAMESV1_H
