#ifndef PYPDSDATA_DETINFO_H
#define PYPDSDATA_DETINFO_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DetInfo.
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
#include "pdsdata/xtc/DetInfo.hh"

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

struct DetInfo {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new DetInfo object from Pds type
  static PyObject* DetInfo_FromPds(const Pds::DetInfo& src);

  // standard Python stuff
  PyObject_HEAD

  Pds::DetInfo m_src;

};

} // namespace pypdsdata

#endif // PYPDSDATA_DETINFO_H
