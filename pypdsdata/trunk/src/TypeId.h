#ifndef PYPDSDATA_TYPEID_H
#define PYPDSDATA_TYPEID_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TypeId.
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
#include "pdsdata/xtc/TypeId.hh"

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

struct TypeId {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  /// Builds TypeId from corresponding Pds type
  static PyObject* TypeId_FromPds(Pds::TypeId type);

  // standard Python stuff
  PyObject_HEAD

  Pds::TypeId m_typeId;

};

} // namespace pypdsdata

#endif // PYPDSDATA_TYPEID_H
