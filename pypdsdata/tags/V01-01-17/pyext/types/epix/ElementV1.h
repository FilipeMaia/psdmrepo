#ifndef PYPDSDATA_EPIX_ELEMENTV1_H
#define PYPDSDATA_EPIX_ELEMENTV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ElementV1.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "../PdsDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/psddl/epix.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {
namespace Epix {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class ElementV1 : public PdsDataType<ElementV1, Pds::Epix::ElementV1> {
public:

  typedef PdsDataType<ElementV1, Pds::Epix::ElementV1> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Epix
} // namespace pypdsdata

#endif // PYPDSDATA_EPIX_ELEMENTV1_H
