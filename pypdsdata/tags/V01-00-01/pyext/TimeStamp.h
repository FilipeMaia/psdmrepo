#ifndef PYPDSDATA_TIMESTAMP_H
#define PYPDSDATA_TIMESTAMP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeStamp.
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
#include "pdsdata/xtc/TimeStamp.hh"

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

class TimeStamp : public PdsDataTypeEmbedded<TimeStamp,Pds::TimeStamp> {
public:

  typedef PdsDataTypeEmbedded<TimeStamp,Pds::TimeStamp> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

namespace Pds {
inline PyObject* toPython(const Pds::TimeStamp& v) { return pypdsdata::TimeStamp::PyObject_FromPds(v); }
}

#endif // PYPDSDATA_TIMESTAMP_H
