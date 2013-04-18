#ifndef PYPDSDATA_EPICS_PVCONFIGV1_H
#define PYPDSDATA_EPICS_PVCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class PvConfigV1.
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
#include "pdsdata/epics/ConfigV1.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Epics {

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

class PvConfigV1 : public PdsDataType<PvConfigV1, Pds::Epics::PvConfigV1> {
public:

  typedef PdsDataType<PvConfigV1, Pds::Epics::PvConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Epics
} // namespace pypdsdata

#endif // PYPDSDATA_EPICS_PVCONFIGV1_H
