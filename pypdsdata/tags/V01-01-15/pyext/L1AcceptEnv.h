#ifndef PYPDSDATA_L1ACCEPTENV_H
#define PYPDSDATA_L1ACCEPTENV_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class L1AcceptEnv.
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
#include "pdsdata/xtc/L1AcceptEnv.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  @brief Python wrapper for Pds::L1AcceptEnv.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class L1AcceptEnv : public PdsDataTypeEmbedded<L1AcceptEnv, Pds::L1AcceptEnv> {
public:

  typedef PdsDataTypeEmbedded<L1AcceptEnv, Pds::L1AcceptEnv> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Dump object info to a stream
  void print(std::ostream& out) const;

};

} // namespace pypdsdata

namespace Pds {
inline PyObject* toPython(const Pds::L1AcceptEnv& v) { return pypdsdata::L1AcceptEnv::PyObject_FromPds(v); }
}

#endif // PYPDSDATA_L1ACCEPTENV_H
