
// *** Do not edit this file, it is auto-generated ***

#include <cstddef>
#include <psddl_psana/lusi.ddl.h> // inc_psana
#include <psddl_python/lusi.ddl.wrapper.h> // inc_python
namespace Psana {
namespace Lusi {
using namespace boost::python;

void createWrappers() {

#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("base", &n::base)\
    .def("scale", &n::scale)\
    .def("_sizeof", &n::_sizeof)\

  _CLASS(Psana::Lusi::DiodeFexConfigV1, return_value_policy<copy_const_reference>());
  _CLASS(Psana::Lusi::DiodeFexConfigV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(DiodeFexConfigV1);
  std_vector_class_(DiodeFexConfigV1_Wrapper);
#undef _CLASS
  ADD_GETTER(DiodeFexConfigV1);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("base", &n::base)\
    .def("scale", &n::scale)\
    .def("_sizeof", &n::_sizeof)\

  _CLASS(Psana::Lusi::DiodeFexConfigV2, return_value_policy<copy_const_reference>());
  _CLASS(Psana::Lusi::DiodeFexConfigV2_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(DiodeFexConfigV2);
  std_vector_class_(DiodeFexConfigV2_Wrapper);
#undef _CLASS
  ADD_GETTER(DiodeFexConfigV2);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("value", &n::value)\
    .def("_sizeof", &n::_sizeof)\

  _CLASS(Psana::Lusi::DiodeFexV1, return_value_policy<copy_const_reference>());
  _CLASS(Psana::Lusi::DiodeFexV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(DiodeFexV1);
  std_vector_class_(DiodeFexV1_Wrapper);
#undef _CLASS


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("diode", &n::diode)\
    .def("xscale", &n::xscale)\
    .def("yscale", &n::yscale)\

  _CLASS(Psana::Lusi::IpmFexConfigV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(IpmFexConfigV1_Wrapper);
#undef _CLASS
  ADD_GETTER(IpmFexConfigV1);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("diode", &n::diode)\
    .def("xscale", &n::xscale)\
    .def("yscale", &n::yscale)\

  _CLASS(Psana::Lusi::IpmFexConfigV2_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(IpmFexConfigV2_Wrapper);
#undef _CLASS
  ADD_GETTER(IpmFexConfigV2);


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("channel", &n::channel)\
    .def("sum", &n::sum)\
    .def("xpos", &n::xpos)\
    .def("ypos", &n::ypos)\
    .def("_sizeof", &n::_sizeof)\

  _CLASS(Psana::Lusi::IpmFexV1, return_value_policy<copy_const_reference>());
  _CLASS(Psana::Lusi::IpmFexV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(IpmFexV1);
  std_vector_class_(IpmFexV1_Wrapper);
#undef _CLASS


#define _CLASS(n, policy) class_<n>(#n, no_init)\
    .def("xscale", &n::xscale)\
    .def("yscale", &n::yscale)\
    .def("_sizeof", &n::_sizeof)\

  _CLASS(Psana::Lusi::PimImageConfigV1, return_value_policy<copy_const_reference>());
  _CLASS(Psana::Lusi::PimImageConfigV1_Wrapper, return_value_policy<return_by_value>());
  std_vector_class_(PimImageConfigV1);
  std_vector_class_(PimImageConfigV1_Wrapper);
#undef _CLASS
  ADD_GETTER(PimImageConfigV1);


} // createWrappers()
} // namespace Lusi
} // namespace Psana
