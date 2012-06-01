
// *** Do not edit this file, it is auto-generated ***

#include <cstddef>
#include <psddl_psana/encoder.ddl.h> // inc_psana
#include <psddl_python/encoder.ddl.wrapper.h> // inc_python
namespace Psana {
namespace Encoder {
using namespace boost::python;

void createWrappers() {

#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("chan_num", &n::chan_num)\
    .def("count_mode", &n::count_mode)\
    .def("quadrature_mode", &n::quadrature_mode)\
    .def("input_num", &n::input_num)\
    .def("input_rising", &n::input_rising)\
    .def("ticks_per_sec", &n::ticks_per_sec)\

  _CLASS(Psana::Encoder::ConfigV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(ConfigV1_Wrapper);
#undef _CLASS
  ADD_GETTER(ConfigV1);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("chan_mask", &n::chan_mask)\
    .def("count_mode", &n::count_mode)\
    .def("quadrature_mode", &n::quadrature_mode)\
    .def("input_num", &n::input_num)\
    .def("input_rising", &n::input_rising)\
    .def("ticks_per_sec", &n::ticks_per_sec)\

  _CLASS(Psana::Encoder::ConfigV2_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(ConfigV2_Wrapper);
#undef _CLASS
  ADD_GETTER(ConfigV2);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("timestamp", &n::timestamp)\
    .def("encoder_count", &n::encoder_count)\

  _CLASS(Psana::Encoder::DataV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(DataV1_Wrapper);
#undef _CLASS
  ADD_GETTER(DataV1);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("timestamp", &n::timestamp)\
    .def("encoder_count", &n::encoder_count)\

  _CLASS(Psana::Encoder::DataV2_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(DataV2_Wrapper);
#undef _CLASS
  ADD_GETTER(DataV2);


} // createWrappers()
} // namespace Encoder
} // namespace Psana
