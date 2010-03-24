#ifndef PYPDSDATA_EPICSPVTIME_H
#define PYPDSDATA_EPICSPVTIME_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpicsPvTime.
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

class EpicsPvTime : public PdsDataType<EpicsPvTime,Pds::EpicsPvHeader> {
public:

  typedef PdsDataType<EpicsPvTime,Pds::EpicsPvHeader> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

#endif // PYPDSDATA_EPICSPVTIME_H
