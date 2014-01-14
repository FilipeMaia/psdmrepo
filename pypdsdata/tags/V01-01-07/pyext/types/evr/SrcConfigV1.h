#ifndef PYPDSDATA_EVRDATA_SRCCONFIGV1_H
#define PYPDSDATA_EVRDATA_SRCCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_SrcConfigV1.
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
#include "pdsdata/psddl/evr.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace EvrData {

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

class SrcConfigV1 : public PdsDataType<SrcConfigV1,Pds::EvrData::SrcConfigV1> {
public:

  typedef PdsDataType<SrcConfigV1,Pds::EvrData::SrcConfigV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace EvrData
} // namespace pypdsdata

#endif // PYPDSDATA_EVRDATA_SRCCONFIGV1_H
