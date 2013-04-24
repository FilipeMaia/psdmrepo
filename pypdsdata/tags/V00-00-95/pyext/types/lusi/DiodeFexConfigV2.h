#ifndef PYPDSDATA_LUSI_DIODEFEXCONFIGV2_H
#define PYPDSDATA_LUSI_DIODEFEXCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class DiodeFexConfigV2.
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
#include "pdsdata/lusi/DiodeFexConfigV2.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Lusi {

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

class DiodeFexConfigV2 : public PdsDataType<DiodeFexConfigV2,Pds::Lusi::DiodeFexConfigV2> {
public:

  typedef PdsDataType<DiodeFexConfigV2,Pds::Lusi::DiodeFexConfigV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Lusi
} // namespace pypdsdata

#endif // PYPDSDATA_LUSI_DIODEFEXCONFIGV2_H
