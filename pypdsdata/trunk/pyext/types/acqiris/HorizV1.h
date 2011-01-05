#ifndef PYPDSDATA_ACQIRIS_HORIZV1_H
#define PYPDSDATA_ACQIRIS_HORIZV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_HorizV1.
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
#include "pdsdata/acqiris/ConfigV1.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Acqiris {

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

class HorizV1 : public PdsDataType<HorizV1,Pds::Acqiris::HorizV1> {
public:

  typedef PdsDataType<HorizV1,Pds::Acqiris::HorizV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;
};

} // namespace Acqiris
} // namespace pypdsdata

#endif // PYPDSDATA_ACQIRIS_HORIZV1_H
