#ifndef PYPDSDATA_ACQIRIS_VERTV1_H
#define PYPDSDATA_ACQIRIS_VERTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_VertV1.
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

class VertV1 : public PdsDataType<VertV1,Pds::Acqiris::VertV1> {
public:

  typedef PdsDataType<VertV1,Pds::Acqiris::VertV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace Acqiris
} // namespace pypdsdata

#endif // PYPDSDATA_ACQIRIS_VERTV1_H
