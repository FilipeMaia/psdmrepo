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
#include "Python.h"

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

/**
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

struct TimeStamp {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new ClockTime object from Pds type
  static PyObject* TimeStamp_FromPds(const Pds::TimeStamp& ts);

  // standard Python stuff
  PyObject_HEAD

  Pds::TimeStamp m_ts;

};

} // namespace pypdsdata

#endif // PYPDSDATA_TIMESTAMP_H
