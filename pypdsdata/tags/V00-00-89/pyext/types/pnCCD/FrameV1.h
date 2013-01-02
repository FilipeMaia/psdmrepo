#ifndef PYPDSDATA_PNCCD_FRAMEV1_H
#define PYPDSDATA_PNCCD_FRAMEV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FrameV1.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/pnCCD/FrameV1.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace PNCCD {

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

class FrameV1 : public PdsDataType<FrameV1,Pds::PNCCD::FrameV1> {
public:

  typedef PdsDataType<FrameV1,Pds::PNCCD::FrameV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace PNCCD
} // namespace pypdsdata

#endif // PYPDSDATA_PNCCD_FRAMEV1_H
