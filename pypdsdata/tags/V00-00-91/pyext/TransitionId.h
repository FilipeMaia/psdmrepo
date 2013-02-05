#ifndef PYPDSDATA_TRANSITIONID_H
#define PYPDSDATA_TRANSITIONID_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TransitionId.
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
#include "pdsdata/xtc/TransitionId.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

/**
 *  be commented with C++-style // (double forward slash) comments.
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
struct TransitionId : PyIntObject {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new TransitionId object from a number
  static PyObject* TransitionId_FromInt(int value);

};

} // namespace pypdsdata

#endif // PYPDSDATA_TRANSITIONID_H
