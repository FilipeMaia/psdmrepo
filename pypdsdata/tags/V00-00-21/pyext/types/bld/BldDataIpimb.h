#ifndef PYPDSDATA_BLD_BLDDATAIPIMB_H
#define PYPDSDATA_BLD_BLDDATAIPIMB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: BldDataIpimb.h 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class BldDataIpimb.
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
#include "pdsdata/bld/bldData.hh"

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
 *  @version $Id: BldDataIpimb.h 811 2010-03-26 17:40:08Z salnikov $
 *
 *  @author Andrei Salnikov
 */

class BldDataIpimb : public PdsDataType<BldDataIpimb,Pds::BldDataIpimb> {
public:

  typedef PdsDataType<BldDataIpimb,Pds::BldDataIpimb> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

#endif // PYPDSDATA_BLD_BLDDATAIPIMB_H
