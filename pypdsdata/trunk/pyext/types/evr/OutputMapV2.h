#ifndef PYPDSDATA_EVRDATA_OUTPUTMAPV2_H
#define PYPDSDATA_EVRDATA_OUTPUTMAPV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OutputMapV2.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../PdsDataTypeEmbedded.h"

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

class EnumType;

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

class OutputMapV2 : public PdsDataTypeEmbedded<OutputMapV2,Pds::EvrData::OutputMapV2> {
public:

  typedef PdsDataTypeEmbedded<OutputMapV2,Pds::EvrData::OutputMapV2> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );
  
  /// access to Conn enum type
  static pypdsdata::EnumType& connEnum() ;

};

} // namespace EvrData
} // namespace pypdsdata

namespace Pds {
namespace EvrData {
inline PyObject* toPython(const Pds::EvrData::OutputMapV2& v) { return pypdsdata::EvrData::OutputMapV2::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_EVRDATA_OUTPUTMAPV2_H
