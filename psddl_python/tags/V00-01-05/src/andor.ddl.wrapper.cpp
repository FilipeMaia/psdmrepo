/* Do not edit this file, as it is auto-generated */

#include <psddl_python/andor.ddl.wrapper.h> // inc_python
#include <cstddef>

namespace psddl_python {
namespace Andor {

void createWrappers() {
  _import_array();

#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("width", &n::width)\
    .def("height", &n::height)\
    .def("orgX", &n::orgX)\
    .def("orgY", &n::orgY)\
    .def("binX", &n::binX)\
    .def("binY", &n::binY)\
    .def("exposureTime", &n::exposureTime)\
    .def("coolingTemp", &n::coolingTemp)\
    .def("fanMode", &n::fanMode)\
    .def("baselineClamp", &n::baselineClamp)\
    .def("highCapacity", &n::highCapacity)\
    .def("gainIndex", &n::gainIndex)\
    .def("readoutSpeedIndex", &n::readoutSpeedIndex)\
    .def("exposureEventCode", &n::exposureEventCode)\
    .def("numDelayShots", &n::numDelayShots)\
    .def("frameSize", &n::frameSize)\
    .def("numPixelsX", &n::numPixelsX)\
    .def("numPixelsY", &n::numPixelsY)\
    .def("numPixels", &n::numPixels)\

  _CLASS(psddl_python::Andor::ConfigV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(ConfigV1_Wrapper);
#undef _CLASS
  ADD_ENV_OBJECT_STORE_GETTER(ConfigV1);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("shotIdStart", &n::shotIdStart)\
    .def("readoutTime", &n::readoutTime)\
    .def("temperature", &n::temperature)\
    .def("data", &n::data)\

  _CLASS(psddl_python::Andor::FrameV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(FrameV1_Wrapper);
#undef _CLASS
  ADD_EVENT_GETTER(FrameV1);


} // createWrappers()
} // namespace Andor
} // namespace psddl_python
