#ifndef PYPDSDATA_BLD_BLDDATAFEEGASDETENERGY_H
#define PYPDSDATA_BLD_BLDDATAFEEGASDETENERGY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataFEEGasDetEnergy.
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
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class BldDataFEEGasDetEnergy : public PdsDataType<BldDataFEEGasDetEnergy,Pds::BldDataFEEGasDetEnergy> {
public:

  typedef PdsDataType<BldDataFEEGasDetEnergy,Pds::BldDataFEEGasDetEnergy> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

#endif // PYPDSDATA_BLD_BLDDATAFEEGASDETENERGY_H
