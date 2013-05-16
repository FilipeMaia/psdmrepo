#ifndef PYPDSDATA_IMP_LANESTATUS_H
#define PYPDSDATA_IMP_LANESTATUS_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LaneStatus.
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
#include "pdsdata/imp/ElementHeader.hh"

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

class LaneStatus : public PdsDataTypeEmbedded<LaneStatus, Pds::Imp::LaneStatus> {
public:

  typedef PdsDataTypeEmbedded<LaneStatus, Pds::Imp::LaneStatus> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& out) const;

};

} // namespace Imp
} // namespace pypdsdata

#endif // PYPDSDATA_IMP_LANESTATUS_H
