#ifndef PYPDSDATA_CSPAD2X2_CSPAD2X2READONLYCFG_H
#define PYPDSDATA_CSPAD2X2_CSPAD2X2READONLYCFG_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ReadOnlyCfg.
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
#include "pdsdata/psddl/cspad2x2.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

namespace pypdsdata {
namespace CsPad2x2 {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class CsPad2x2ReadOnlyCfg : public PdsDataTypeEmbedded<CsPad2x2ReadOnlyCfg, Pds::CsPad2x2::CsPad2x2ReadOnlyCfg> {
public:

  typedef PdsDataTypeEmbedded<CsPad2x2ReadOnlyCfg, Pds::CsPad2x2::CsPad2x2ReadOnlyCfg> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace CsPad2x2
} // namespace pypdsdata

namespace Pds {
namespace CsPad2x2 {
inline PyObject* toPython(const Pds::CsPad2x2::CsPad2x2ReadOnlyCfg& v) { return pypdsdata::CsPad2x2::CsPad2x2ReadOnlyCfg::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_CSPAD2X2_CSPAD2X2READONLYCFG_H
