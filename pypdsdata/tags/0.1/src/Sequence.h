#ifndef PYPDSDATA_SEQUENCE_H
#define PYPDSDATA_SEQUENCE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Sequence.
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
#include "pdsdata/xtc/Sequence.hh"

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

struct Sequence  {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new Sequence object from Pds type
  static PyObject* Sequence_FromPds(const Pds::Sequence& seq);

  // standard Python stuff
  PyObject_HEAD

  Pds::Sequence m_seq;

};

} // namespace pypdsdata

#endif // PYPDSDATA_SEQUENCE_H
