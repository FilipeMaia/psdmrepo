#ifndef PYPDSDATA_DAMAGE_H
#define PYPDSDATA_DAMAGE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Damage.
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
#include "pdsdata/xtc/Damage.hh"

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

struct Damage : PyIntObject {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new TransitionId object from a number
  static PyObject* Damage_FromInt(int value);

};

} // namespace pypdsdata

#endif // PYPDSDATA_DAMAGE_H
