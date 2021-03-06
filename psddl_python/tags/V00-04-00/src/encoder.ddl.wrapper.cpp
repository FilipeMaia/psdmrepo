/* Do not edit this file, as it is auto-generated */

#include <boost/make_shared.hpp>
#include "psddl_python/encoder.ddl.wrapper.h" // inc_python
#include "psddl_python/ConverterMap.h"

namespace psddl_python {
namespace Encoder {

namespace {
PyObject* method_typeid_ConfigV1() {
  static PyObject* ptypeid = PyCObject_FromVoidPtr((void*)&typeid(Psana::Encoder::ConfigV1), 0);
  Py_INCREF(ptypeid);
  return ptypeid;
}

PyObject* method_typeid_ConfigV2() {
  static PyObject* ptypeid = PyCObject_FromVoidPtr((void*)&typeid(Psana::Encoder::ConfigV2), 0);
  Py_INCREF(ptypeid);
  return ptypeid;
}

PyObject* method_typeid_DataV1() {
  static PyObject* ptypeid = PyCObject_FromVoidPtr((void*)&typeid(Psana::Encoder::DataV1), 0);
  Py_INCREF(ptypeid);
  return ptypeid;
}

PyObject* method_typeid_DataV2() {
  static PyObject* ptypeid = PyCObject_FromVoidPtr((void*)&typeid(Psana::Encoder::DataV2), 0);
  Py_INCREF(ptypeid);
  return ptypeid;
}

} // namespace
void createWrappers(PyObject* module) {
  PyObject* submodule = Py_InitModule3( "psana.Encoder", 0, "The Python wrapper module for Encoder types");
  Py_INCREF(submodule);
  PyModule_AddObject(module, "Encoder", submodule);
  scope mod = object(handle<>(borrowed(submodule)));
  class_<psddl_python::Encoder::ConfigV1_Wrapper>("ConfigV1", no_init)
    .def("chan_num", &psddl_python::Encoder::ConfigV1_Wrapper::chan_num)
    .def("count_mode", &psddl_python::Encoder::ConfigV1_Wrapper::count_mode)
    .def("quadrature_mode", &psddl_python::Encoder::ConfigV1_Wrapper::quadrature_mode)
    .def("input_num", &psddl_python::Encoder::ConfigV1_Wrapper::input_num)
    .def("input_rising", &psddl_python::Encoder::ConfigV1_Wrapper::input_rising)
    .def("ticks_per_sec", &psddl_python::Encoder::ConfigV1_Wrapper::ticks_per_sec)
    .def("__typeid__", &method_typeid_ConfigV1)
    .staticmethod("__typeid__")
  ;
  psddl_python::ConverterMap::instance().addConverter(boost::make_shared<ConfigV1_Converter>());

  class_<psddl_python::Encoder::ConfigV2_Wrapper>("ConfigV2", no_init)
    .def("chan_mask", &psddl_python::Encoder::ConfigV2_Wrapper::chan_mask)
    .def("count_mode", &psddl_python::Encoder::ConfigV2_Wrapper::count_mode)
    .def("quadrature_mode", &psddl_python::Encoder::ConfigV2_Wrapper::quadrature_mode)
    .def("input_num", &psddl_python::Encoder::ConfigV2_Wrapper::input_num)
    .def("input_rising", &psddl_python::Encoder::ConfigV2_Wrapper::input_rising)
    .def("ticks_per_sec", &psddl_python::Encoder::ConfigV2_Wrapper::ticks_per_sec)
    .def("__typeid__", &method_typeid_ConfigV2)
    .staticmethod("__typeid__")
  ;
  psddl_python::ConverterMap::instance().addConverter(boost::make_shared<ConfigV2_Converter>());

  class_<psddl_python::Encoder::DataV1_Wrapper>("DataV1", no_init)
    .def("timestamp", &psddl_python::Encoder::DataV1_Wrapper::timestamp)
    .def("encoder_count", &psddl_python::Encoder::DataV1_Wrapper::encoder_count)
    .def("value", &psddl_python::Encoder::DataV1_Wrapper::value)
    .def("__typeid__", &method_typeid_DataV1)
    .staticmethod("__typeid__")
  ;
  psddl_python::ConverterMap::instance().addConverter(boost::make_shared<DataV1_Converter>());

  class_<psddl_python::Encoder::DataV2_Wrapper>("DataV2", no_init)
    .def("timestamp", &psddl_python::Encoder::DataV2_Wrapper::timestamp)
    .def("encoder_count", &psddl_python::Encoder::DataV2_Wrapper::encoder_count)
    .def("value", &psddl_python::Encoder::DataV2_Wrapper::value)
    .def("__typeid__", &method_typeid_DataV2)
    .staticmethod("__typeid__")
  ;
  psddl_python::ConverterMap::instance().addConverter(boost::make_shared<DataV2_Converter>());

  {
    PyObject* unvlist = PyList_New(2);
    PyList_SET_ITEM(unvlist, 0, PyObject_GetAttrString(submodule, "DataV1"));
    PyList_SET_ITEM(unvlist, 1, PyObject_GetAttrString(submodule, "DataV2"));
    PyObject_SetAttrString(submodule, "Data", unvlist);
    Py_CLEAR(unvlist);
  }
  {
    PyObject* unvlist = PyList_New(2);
    PyList_SET_ITEM(unvlist, 0, PyObject_GetAttrString(submodule, "ConfigV1"));
    PyList_SET_ITEM(unvlist, 1, PyObject_GetAttrString(submodule, "ConfigV2"));
    PyObject_SetAttrString(submodule, "Config", unvlist);
    Py_CLEAR(unvlist);
  }
  detail::register_ndarray_to_numpy_cvt<const uint32_t, 1>();

} // createWrappers()
} // namespace Encoder
} // namespace psddl_python
