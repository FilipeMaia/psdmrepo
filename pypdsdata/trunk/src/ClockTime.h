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
#include "Python.h"

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

/**
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

struct ClockTime {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new ClockTime object from Pds type
  static PyObject* ClockTime_FromPds(const Pds::ClockTime& clock);

  // standard Python stuff
  PyObject_HEAD

  Pds::ClockTime m_clock;

};

} // namespace pypdsdata

#endif // PYPDSDATA_CLOCKTIME_H
