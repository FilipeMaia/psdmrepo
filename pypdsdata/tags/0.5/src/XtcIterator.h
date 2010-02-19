#ifndef PYPDSDATA_XTCITERATOR_H
#define PYPDSDATA_XTCITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcIterator.
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
#include "pdsdata/xtc/XtcIterator.hh"

//    ---------------------
//    -- Class Interface --
//    ---------------------

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

struct XtcIterator {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  /// Builds iterator from Xtc.
  static PyObject* XtcIterator_FromXtc( Pds::Xtc* xtc, PyObject* parent );

  // standard Python stuff
  PyObject_HEAD

  PyObject* m_parent;
  Pds::Xtc* m_parentXtc;
  Pds::Xtc* m_next;
  int m_remaining ;

};

} // namespace pypdsdata

#endif // PYPDSDATA_XTCITERATOR_H
