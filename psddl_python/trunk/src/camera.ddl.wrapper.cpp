/* Do not edit this file, as it is auto-generated */

#include <cstddef>
#include <psddl_psana/camera.ddl.h> // inc_psana
#include <psddl_python/camera.ddl.wrapper.h> // inc_python

namespace psddl_python {
namespace Camera {

void createWrappers() {
  _import_array();

#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("column", &n::column)\
    .def("row", &n::row)\
    .def("_sizeof", &n::_sizeof)\

  _CLASS(Psana::Camera::FrameCoord, return_value_policy<copy_const_reference>());
  _CLASS(psddl_python::Camera::FrameCoord_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(Psana::Camera::FrameCoord);
  std_vector_class_(FrameCoord_Wrapper);
#undef _CLASS
  ADD_EVENT_GETTER(FrameCoord);


#define _CLASS(n, policy) class_<n>(#n, no_init)\

  _CLASS(psddl_python::Camera::FrameFccdConfigV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(FrameFccdConfigV1_Wrapper);
#undef _CLASS
  ADD_ENV_OBJECT_STORE_GETTER(FrameFccdConfigV1);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("forwarding", &n::forwarding)\
    .def("forward_prescale", &n::forward_prescale)\
    .def("processing", &n::processing)\
    .def("roiBegin", &n::roiBegin, policy)\
    .def("roiEnd", &n::roiEnd, policy)\
    .def("threshold", &n::threshold)\
    .def("number_of_masked_pixels", &n::number_of_masked_pixels)\
    .def("masked_pixel_coordinates", &n::masked_pixel_coordinates)\

  _CLASS(psddl_python::Camera::FrameFexConfigV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(FrameFexConfigV1_Wrapper);
#undef _CLASS
  ADD_ENV_OBJECT_STORE_GETTER(FrameFexConfigV1);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("width", &n::width)\
    .def("height", &n::height)\
    .def("depth", &n::depth)\
    .def("offset", &n::offset)\
    .def("_int_pixel_data", &n::_int_pixel_data)\
    .def("data8", &n::data8)\
    .def("data16", &n::data16)\

  _CLASS(psddl_python::Camera::FrameV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(FrameV1_Wrapper);
#undef _CLASS
  ADD_EVENT_GETTER(FrameV1);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("integral", &n::integral)\
    .def("xmean", &n::xmean)\
    .def("ymean", &n::ymean)\
    .def("major_axis_width", &n::major_axis_width)\
    .def("minor_axis_width", &n::minor_axis_width)\
    .def("major_axis_tilt", &n::major_axis_tilt)\

  _CLASS(psddl_python::Camera::TwoDGaussianV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(TwoDGaussianV1_Wrapper);
#undef _CLASS
  ADD_EVENT_GETTER(TwoDGaussianV1);


} // createWrappers()
} // namespace Camera
} // namespace psddl_python
