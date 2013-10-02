#ifndef PYPDSDATA_TYPEID_H
#define PYPDSDATA_TYPEID_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TypeId.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "types/PdsDataTypeEmbedded.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/xtc/TypeId.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

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

class TypeId : public PdsDataTypeEmbedded<TypeId,Pds::TypeId> {
public:

  typedef PdsDataTypeEmbedded<TypeId,Pds::TypeId> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Dump object info to a stream
  void print(std::ostream& out) const;
};

} // namespace pypdsdata

namespace Pds {
inline PyObject* toPython(const Pds::TypeId& v) { return pypdsdata::TypeId::PyObject_FromPds(v); }
}

#endif // PYPDSDATA_TYPEID_H
