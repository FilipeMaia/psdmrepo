#ifndef PSANA_PYTHON_PYEXT_SCAN_H
#define PSANA_PYTHON_PYEXT_SCAN_H

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
#include "pytools/PyDataType.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "psana/Scan.h"

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace psana_python {
namespace pyext {

/**
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class Scan : public pytools::PyDataType<Scan, psana::Scan> {
public:

  typedef pytools::PyDataType<Scan, psana::Scan> BaseType;

  /// Initialize Python type and register it in a module
  static void initType( PyObject* module );

};

} // namespace pyext
} // namespace psana

#endif // PSANA_PYTHON_PYEXT_SCAN_H
