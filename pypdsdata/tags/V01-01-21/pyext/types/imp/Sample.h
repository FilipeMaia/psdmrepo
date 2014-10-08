#ifndef PYPDSDATA_IMP_SAMPLE_H
#define PYPDSDATA_IMP_SAMPLE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Sample.
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
#include "pdsdata/psddl/imp.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace Imp {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Sample : public PdsDataTypeEmbedded<Sample, Pds::Imp::Sample> {
public:

  typedef PdsDataTypeEmbedded<Sample, Pds::Imp::Sample> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;

};

} // namespace Imp
} // namespace pypdsdata

namespace Pds {
namespace Imp {
inline PyObject* toPython(const Pds::Imp::Sample& v) { return pypdsdata::Imp::Sample::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_IMP_SAMPLE_H
