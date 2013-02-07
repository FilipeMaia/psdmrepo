#ifndef PYTOOLS_MAKE_PYSHARED_H
#define PYTOOLS_MAKE_PYSHARED_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class make_pyshared.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>
#include "python/Python.h"

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace pytools {

namespace detail {
// special deleter for shared_ptr to use with PyObject
struct py_obj_delete {
  void operator()(PyObject* obj) { Py_CLEAR(obj); }
};
} // namespace detail

typedef boost::shared_ptr<PyObject> pyshared_ptr;

/// @addtogroup pytools

/**
 *  @ingroup pytools
 *
 *  @brief Factory function which creates shared pointers for Python objects.
 *  
 *  By default this method "steals" pointer to the Python object and returns a shared 
 *  pointer object which owns that object. This is good behavior for new object 
 *  references. If optional second parameter is set to false then it is assumed that
 *  reference is borrowed and function increments reference counter for it first.  
 *  
 *  When last copy of shared pointed disappears the reference counter for Python object 
 *  will be decremented.
 *  
 *  @param[in] object  Pointer to an existing Python object
 *  @param[in] steal   If set to true then object is assumed to be a new reference,
 *                     otherwise it should be a borrowed reference.
 */
inline
pyshared_ptr
make_pyshared(PyObject* object, bool steal = true) { 
  if (not steal) Py_INCREF(object);
  return pyshared_ptr(object, detail::py_obj_delete());
}

} // namespace pytools

#endif // PYTOOLS_MAKE_PYSHARED_H
