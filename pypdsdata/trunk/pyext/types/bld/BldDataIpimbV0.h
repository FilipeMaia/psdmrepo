#ifndef PYPDSDATA_BLD_BLDDATAIPIMBV0_H
#define PYPDSDATA_BLD_BLDDATAIPIMBV0_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: BldDataIpimbV0.h 811 2010-03-26 17:40:08Z salnikov $
//
// Description:
//	Class BldDataIpimbV0.
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
 *  @version $Id: BldDataIpimbV0.h 811 2010-03-26 17:40:08Z salnikov $
 *
 *  @author Andrei Salnikov
 */

class BldDataIpimbV0 : public PdsDataType<BldDataIpimbV0,Pds::BldDataIpimbV0> {
public:

  typedef PdsDataType<BldDataIpimbV0,Pds::BldDataIpimbV0> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

#endif // PYPDSDATA_BLD_BLDDATAIPIMBV0_H
