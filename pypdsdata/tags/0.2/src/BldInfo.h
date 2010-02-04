#ifndef PYPDSDATA_BLDINFO_H
#define PYPDSDATA_BLDINFO_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldInfo.
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
#include "pdsdata/xtc/BldInfo.hh"

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

struct BldInfo {

  /// Returns the Python type
  static PyTypeObject* typeObject();

  // makes new BldInfo object from Pds type
  static PyObject* BldInfo_FromPds(const Pds::BldInfo& src);

  // standard Python stuff
  PyObject_HEAD

  Pds::BldInfo m_src;

};

} // namespace pypdsdata

#endif // PYPDSDATA_BLDINFO_H
