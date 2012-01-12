#ifndef PYPDSDATA_LUSI_DIODEFEXV1_H
#define PYPDSDATA_LUSI_DIODEFEXV1_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id: DiodeFexV1.h 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//      Class DiodeFexV1.
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
#include "pdsdata/lusi/DiodeFexV1.hh"

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
 *  @version $Id: DiodeFexV1.h 811 2010-03-26 17:40:08Z salnikov $
 *
 *  @author Andrei Salnikov
 */

class DiodeFexV1 : public PdsDataType<DiodeFexV1,Pds::Lusi::DiodeFexV1> {
public:

  typedef PdsDataType<DiodeFexV1,Pds::Lusi::DiodeFexV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Lusi
} // namespace pypdsdata

#endif // PYPDSDATA_LUSI_DIODEFEXV1_H
