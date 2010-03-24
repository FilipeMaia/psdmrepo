#ifndef PYPDSDATA_PULNIX_TM6740CONFIGV1_H
#define PYPDSDATA_PULNIX_TM6740CONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Pulnix_TM6740ConfigV1.
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
#include "pdsdata/pulnix/TM6740ConfigV1.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Pulnix {

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

class TM6740ConfigV1 : public PdsDataType<TM6740ConfigV1,Pds::Pulnix::TM6740ConfigV1> {
public:

  typedef PdsDataType<TM6740ConfigV1,Pds::Pulnix::TM6740ConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Pulnix
} // namespace pypdsdata

#endif // PYPDSDATA_PULNIX_TM6740CONFIGV1_H
