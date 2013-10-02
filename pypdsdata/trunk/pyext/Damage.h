#ifndef PYPDSDATA_DAMAGE_H
#define PYPDSDATA_DAMAGE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Damage.
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
#include "pdsdata/xtc/Damage.hh"

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

class Damage : public PdsDataTypeEmbedded<Damage,Pds::Damage> {
public:

  typedef PdsDataTypeEmbedded<Damage,Pds::Damage> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Dump object info to a stream
  void print(std::ostream& out) const;
};

} // namespace pypdsdata

namespace Pds {
inline PyObject* toPython(const Pds::Damage& v) { return pypdsdata::Damage::PyObject_FromPds(v); }
}

#endif // PYPDSDATA_DAMAGE_H
