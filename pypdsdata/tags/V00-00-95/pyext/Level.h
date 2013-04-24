#ifndef PYPDSDATA_LEVEL_H
#define PYPDSDATA_LEVEL_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Level.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "python/Python.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "pdsdata/xtc/Level.hh"

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

struct Level : PyIntObject {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new Level object from a number
  static PyObject* Level_FromInt(int value);

};

} // namespace pypdsdata

#endif // PYPDSDATA_LEVEL_H
