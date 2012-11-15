#ifndef PYPDSDATA_BLD_BLDDATAEBEAMV1_H
#define PYPDSDATA_BLD_BLDDATAEBEAMV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV1.
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
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class BldDataEBeamV1 : public PdsDataType<BldDataEBeamV1,Pds::BldDataEBeamV1> {
public:

  typedef PdsDataType<BldDataEBeamV1,Pds::BldDataEBeamV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace pypdsdata

#endif // PYPDSDATA_BLD_BLDDATAEBEAMV1_H
