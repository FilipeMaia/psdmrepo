#ifndef PYPDSDATA_CSPAD2X2_CSPADPROTECTIONSYSTEMTHRESHOLD_H
#define PYPDSDATA_CSPAD2X2_CSPADPROTECTIONSYSTEMTHRESHOLD_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPadProtectionSystemThreshold.
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

class CsPadProtectionSystemThreshold : public PdsDataTypeEmbedded<CsPadProtectionSystemThreshold, Pds::CsPad2x2::ProtectionSystemThreshold> {
public:

  typedef PdsDataTypeEmbedded<CsPadProtectionSystemThreshold, Pds::CsPad2x2::ProtectionSystemThreshold> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace CsPad2x2
} // namespace pypdsdata

namespace Pds {
namespace CsPad2x2 {
inline PyObject* toPython(const Pds::CsPad2x2::ProtectionSystemThreshold& v) { return pypdsdata::CsPad2x2::CsPadProtectionSystemThreshold::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_CSPAD2X2_CSPADPROTECTIONSYSTEMTHRESHOLD_H
