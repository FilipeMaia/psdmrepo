#ifndef PYPDSDATA_CLOCKTIME_H
#define PYPDSDATA_CLOCKTIME_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ClockTime.
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
#include "pdsdata/xtc/ClockTime.hh"

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
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class ClockTime : public PdsDataTypeEmbedded<ClockTime,Pds::ClockTime> {
public:

  typedef PdsDataTypeEmbedded<ClockTime,Pds::ClockTime> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pypdsdata

namespace Pds {
inline PyObject* toPython(const Pds::ClockTime& v) { return pypdsdata::ClockTime::PyObject_FromPds(v); }
}

#endif // PYPDSDATA_CLOCKTIME_H
