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
#include "types/PdsDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/epics/EpicsPvData.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {

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

class EpicsPvCtrl : public PdsDataType<EpicsPvCtrl,Pds::EpicsPvCtrlHeader> {
public:

  typedef PdsDataType<EpicsPvCtrl,Pds::EpicsPvCtrlHeader> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

#endif // PYPDSDATA_EPICSPVCTRL_H
