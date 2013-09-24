#ifndef PYPDSDATA_EVRDATA_OUTPUTMAP_H
#define PYPDSDATA_EVRDATA_OUTPUTMAP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OutputMap.
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

class OutputMap : public PdsDataTypeEmbedded<OutputMap,Pds::EvrData::OutputMap> {
public:

  typedef PdsDataTypeEmbedded<OutputMap,Pds::EvrData::OutputMap> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );
  
  /// access to Conn enum type
  static pypdsdata::EnumType& connEnum() ;

};

} // namespace EvrData
} // namespace pypdsdata

namespace Pds {
namespace EvrData {
inline PyObject* toPython(const Pds::EvrData::OutputMap& v) { return pypdsdata::EvrData::OutputMap::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_EVRDATA_OUTPUTMAP_H
