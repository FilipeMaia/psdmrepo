#ifndef PYPDSDATA_SRC_H
#define PYPDSDATA_SRC_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Src.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include "python/Python.h"
#include <iosfwd>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pypdsdata {

/// @addtogroup pypdsdata

/**
 *  @ingroup pypdsdata
 *
 *  @brief This is just an utility class with few helper methods.
 *
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Src  {
public:

  /**
   *  Method that knows how to print the content of Src.
   */
  static void print(std::ostream& out, const Pds::Src& src);

  /// convert src to python object
  static PyObject* PyObject_FromPds(const Pds::Src& src);
  
};

} // namespace pypdsdata

namespace Pds {
inline PyObject* toPython(const Pds::Src& v) { return pypdsdata::Src::PyObject_FromPds(v); }
}

#endif // PYPDSDATA_SRC_H
