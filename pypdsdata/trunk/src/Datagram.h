#ifndef PYPDSDATA_DATAGRAM_H
#define PYPDSDATA_DATAGRAM_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Datagram.
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
#include "pdsdata/xtc/Dgram.hh"

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

struct Datagram  {

  // type of the destructor function
  typedef void (*destructor)(Pds::Dgram*);

  /// Returns the Python type
  static PyTypeObject* typeObject();

  /// Builds Datagram from corresponding Pds type, parent is the owner
  /// of the corresponding buffer space, if parent is 0 then datagram
  /// will be deleted on destruction.
  static PyObject* Datagram_FromPds( Pds::Dgram* object, PyObject* parent, destructor dtor );

  // standard Python stuff
  PyObject_HEAD

  Pds::Dgram* m_object;
  PyObject* m_parent;
  destructor m_dtor;

};

} // namespace pypdsdata

#endif // PYPDSDATA_DATAGRAM_H
