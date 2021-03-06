#ifndef PYPDSDATA_EPICSPVCTRL_H
#define PYPDSDATA_EPICSPVCTRL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsPvCtrl.
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
#include "pdsdata/psddl/epics.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Epics {

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

class EpicsPvCtrl : public PdsDataType<EpicsPvCtrl,Pds::Epics::EpicsPvCtrlHeader> {
public:

  typedef PdsDataType<EpicsPvCtrl,Pds::Epics::EpicsPvCtrlHeader> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Epics
} // namespace pypdsdata

#endif // PYPDSDATA_EPICSPVCTRL_H
