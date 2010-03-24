#ifndef PYPDSDATA_XTCFILEITERATOR_H
#define PYPDSDATA_XTCFILEITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFileIterator.
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
#include "pdsdata/xtc/XtcFileIterator.hh"

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

struct XtcFileIterator : PyObject {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  /// factory function
  static XtcFileIterator* XtcFileIterator_FromFile( PyObject* file );

  PyObject* m_file;
  size_t m_count ;

};

} // namespace pypdsdata

#endif // PYPDSDATA_XTCFILEITERATOR_H
