#ifndef PYPDSDATA_LUSI_DIODEFEXCONFIGV1_H
#define PYPDSDATA_LUSI_DIODEFEXCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id: DiodeFexConfigV1.h 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//      Class DiodeFexConfigV1.
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
#include "pdsdata/psddl/lusi.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Lusi {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id: DiodeFexConfigV1.h 811 2010-03-26 17:40:08Z salnikov $
 *
 *  @author Andrei Salnikov
 */

class DiodeFexConfigV1 : public PdsDataType<DiodeFexConfigV1,Pds::Lusi::DiodeFexConfigV1> {
public:

  typedef PdsDataType<DiodeFexConfigV1,Pds::Lusi::DiodeFexConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Lusi
} // namespace pypdsdata

#endif // PYPDSDATA_LUSI_DIODEFEXCONFIGV1_H
