#include <psddl_python/CreateDeviceWrappers.h>

namespace psddl_python {
#undef DECL
#define DECL(ns) namespace ns { extern void createWrappers(PyObject* module); }
#include "WrapperList.txt"
}

namespace psddl_python {
void createDeviceWrappers(PyObject* module) {

#undef DECL
#define DECL(ns) psddl_python::ns::createWrappers(module);
#include "WrapperList.txt"

}
} // namespace Psana
