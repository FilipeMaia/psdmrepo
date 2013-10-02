#ifndef PYPDSDATA_ALIAS_SRCALIAS_H
#define PYPDSDATA_ALIAS_SRCALIAS_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id$
//
// Description:
//      Class SrcAlias.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/psddl/alias.ddl.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace pypdsdata {
namespace Alias {

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

class SrcAlias : public PdsDataTypeEmbedded<SrcAlias, Pds::Alias::SrcAlias> {
public:

  typedef PdsDataTypeEmbedded<SrcAlias, Pds::Alias::SrcAlias> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

  // dump to a stream
  void print(std::ostream& str) const;

};

} // namespace Alias
} // namespace pypdsdata

namespace Pds {
namespace Alias {
inline PyObject* toPython(const Pds::Alias::SrcAlias& v) { return pypdsdata::Alias::SrcAlias::PyObject_FromPds(v); }
}
}

#endif // PYPDSDATA_ALIAS_SRCALIAS_H
