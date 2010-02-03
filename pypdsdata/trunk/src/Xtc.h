#ifndef PYPDSDATA_XTC_H
#define PYPDSDATA_XTC_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Xtc.
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
#include "pdsdata/xtc/Xtc.hh"

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

struct Xtc {

  // type of the destructor function
  typedef void (*destructor)(Pds::Xtc*);

  /// Returns the Python type
  static PyTypeObject* typeObject();

  /// Builds Xtc from corresponding Pds type, parent is the owner
  /// of the corresponding buffer space, if parent is 0 then Xtc
  /// will be deleted on destruction.
  static PyObject* Xtc_FromPds( Pds::Xtc* xtc, PyObject* parent, destructor dtor );

  // Check object type
  static bool Xtc_Check( PyObject* obj );

  // REturns a pointer to Pds object
  static Pds::Xtc* Xtc_AsPds( PyObject* obj );

  // standard Python stuff
  PyObject_HEAD

  Pds::Xtc* m_xtc;
  PyObject* m_parent;
  destructor m_dtor;

};

} // namespace pypdsdata

#endif // PYPDSDATA_XTC_H
