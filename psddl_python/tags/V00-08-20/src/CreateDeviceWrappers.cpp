#include "psddl_python/CreateDeviceWrappers.h"

#include <boost/python.hpp>

#define PSDDL_PYTHON_IMPORT_ARRAY 1
#include "psddl_python/psddl_python_numpy.h"

namespace psddl_python {
#undef DECL
#define DECL(ns) namespace ns { extern void createWrappers(PyObject* module); }
#include "WrapperList.txt"
#undef DECL
}

namespace psddl_python {
void createDeviceWrappers(PyObject* module) {
  boost::python::docstring_options local_docstring_options(true, true, false);

  // import numpy
  _import_array();
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

#undef DECL
#define DECL(ns) psddl_python::ns::createWrappers(module);
#include "WrapperList.txt"
#undef DECL

}
} // namespace psddl_python
