#ifndef PYPDSDATA_ENV_H
#define PYPDSDATA_ENV_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Env.
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
#include "pdsdata/xtc/Env.hh"

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
 *  @brief Python wrapper class for Pds::Env.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Env : public PdsDataTypeEmbedded<Env, Pds::Env> {
public:

  typedef PdsDataTypeEmbedded<Env, Pds::Env> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // Dump object info to a stream
  void print(std::ostream& out) const;

};

} // namespace pypdsdata

namespace Pds {
inline PyObject* toPython(const Pds::Env& v) { return pypdsdata::Env::PyObject_FromPds(v); }
}

#endif // PYPDSDATA_ENV_H
