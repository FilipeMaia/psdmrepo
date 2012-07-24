/* Do not edit this file, as it is auto-generated */

#include <cstddef>
#include <psddl_psana/fccd.ddl.h> // inc_psana
#include <psddl_python/fccd.ddl.wrapper.h> // inc_python

namespace Psana {
namespace FCCD {

void createWrappers() {

#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("outputMode", &n::outputMode)\
    .def("width", &n::width)\
    .def("height", &n::height)\
    .def("trimmedWidth", &n::trimmedWidth)\
    .def("trimmedHeight", &n::trimmedHeight)\

  _CLASS(Psana::FCCD::FccdConfigV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(FccdConfigV1_Wrapper);
#undef _CLASS
  ADD_ENV_OBJECT_STORE_GETTER(FccdConfigV1);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("outputMode", &n::outputMode)\
    .def("ccdEnable", &n::ccdEnable)\
    .def("focusMode", &n::focusMode)\
    .def("exposureTime", &n::exposureTime)\
    .def("dacVoltages", &n::dacVoltages)\
    .def("waveforms", &n::waveforms)\
    .def("width", &n::width)\
    .def("height", &n::height)\
    .def("trimmedWidth", &n::trimmedWidth)\
    .def("trimmedHeight", &n::trimmedHeight)\

  _CLASS(Psana::FCCD::FccdConfigV2_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(FccdConfigV2_Wrapper);
#undef _CLASS
  ADD_ENV_OBJECT_STORE_GETTER(FccdConfigV2);


} // createWrappers()
} // namespace FCCD
} // namespace Psana
