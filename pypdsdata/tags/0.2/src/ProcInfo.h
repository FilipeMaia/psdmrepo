#ifndef PYPDSDATA_PROCINFO_H
#define PYPDSDATA_PROCINFO_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ProcInfo.
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
#include "pdsdata/xtc/ProcInfo.hh"

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

struct ProcInfo {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new ProcInfo object from Pds type
  static PyObject* ProcInfo_FromPds(const Pds::ProcInfo& src);

  // standard Python stuff
  PyObject_HEAD

  Pds::ProcInfo m_src;

};

} // namespace pypdsdata

#endif // PYPDSDATA_PROCINFO_H
