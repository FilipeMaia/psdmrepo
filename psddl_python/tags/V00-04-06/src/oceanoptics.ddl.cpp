/* Do not edit this file, as it is auto-generated */

#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include "ndarray/ndarray.h"
#include "pdsdata/xtc/TypeId.hh"
#include "psddl_psana/oceanoptics.ddl.h" // inc_psana
#include "psddl_python/Converter.h"
#include "psddl_python/DdlWrapper.h"
#include "psddl_python/ConverterMap.h"
#include "psddl_python/ConverterBoostDef.h"
#include "psddl_python/ConverterBoostDefSharedPtr.h"

namespace psddl_python {
namespace OceanOptics {

using namespace boost::python;
using boost::python::object;
using boost::shared_ptr;
using std::vector;

namespace {
template <typename T>
PyObject* method_typeid() {
  static PyObject* ptypeid = PyCObject_FromVoidPtr((void*)&typeid(T), 0);
  Py_INCREF(ptypeid);
  return ptypeid;
}
template<typename T, std::vector<int> (T::*MF)() const>
PyObject* method_shape(const T *x) {
  return detail::vintToList((x->*MF)());
}
} // namespace

void createWrappers(PyObject* module) {
  PyObject* submodule = Py_InitModule3( "psana.OceanOptics", 0, "The Python wrapper module for OceanOptics types");
  Py_INCREF(submodule);
  PyModule_AddObject(module, "OceanOptics", submodule);
  scope mod = object(handle<>(borrowed(submodule)));
  class_<Psana::OceanOptics::ConfigV1, boost::shared_ptr<Psana::OceanOptics::ConfigV1>, boost::noncopyable >("ConfigV1", no_init)
    .def("exposureTime", &Psana::OceanOptics::ConfigV1::exposureTime)
    .def("waveLenCalib", &Psana::OceanOptics::ConfigV1::waveLenCalib)
    .def("nonlinCorrect", &Psana::OceanOptics::ConfigV1::nonlinCorrect)
    .def("strayLightConstant", &Psana::OceanOptics::ConfigV1::strayLightConstant)
    .def("__typeid__", &method_typeid<Psana::OceanOptics::ConfigV1>)
    .staticmethod("__typeid__")
  ;
  ConverterMap::instance().addConverter(boost::make_shared<ConverterBoostDefSharedPtr<Psana::OceanOptics::ConfigV1> >(Pds::TypeId::Id_OceanOpticsConfig, 1));

  class_<Psana::OceanOptics::timespec64 >("timespec64", no_init)
    .def("tv_sec", &Psana::OceanOptics::timespec64::tv_sec)
    .def("tv_nsec", &Psana::OceanOptics::timespec64::tv_nsec)
    .def("__typeid__", &method_typeid<Psana::OceanOptics::timespec64>)
    .staticmethod("__typeid__")
  ;
  ConverterMap::instance().addConverter(boost::make_shared<ConverterBoostDef<Psana::OceanOptics::timespec64> >(-1, -1));

  class_<Psana::OceanOptics::DataV1, boost::shared_ptr<Psana::OceanOptics::DataV1>, boost::noncopyable >("DataV1", no_init)
    .def("data", &Psana::OceanOptics::DataV1::data)
    .def("frameCounter", &Psana::OceanOptics::DataV1::frameCounter)
    .def("numDelayedFrames", &Psana::OceanOptics::DataV1::numDelayedFrames)
    .def("numDiscardFrames", &Psana::OceanOptics::DataV1::numDiscardFrames)
    .def("timeFrameStart", &Psana::OceanOptics::DataV1::timeFrameStart, return_value_policy<copy_const_reference>())
    .def("timeFrameFirstData", &Psana::OceanOptics::DataV1::timeFrameFirstData, return_value_policy<copy_const_reference>())
    .def("timeFrameEnd", &Psana::OceanOptics::DataV1::timeFrameEnd, return_value_policy<copy_const_reference>())
    .def("version", &Psana::OceanOptics::DataV1::version)
    .def("numSpectraInData", &Psana::OceanOptics::DataV1::numSpectraInData)
    .def("numSpectraInQueue", &Psana::OceanOptics::DataV1::numSpectraInQueue)
    .def("numSpectraUnused", &Psana::OceanOptics::DataV1::numSpectraUnused)
    .def("durationOfFrame", &Psana::OceanOptics::DataV1::durationOfFrame)
    .def("nonlinerCorrected", &Psana::OceanOptics::DataV1::nonlinerCorrected)
    .def("__typeid__", &method_typeid<Psana::OceanOptics::DataV1>)
    .staticmethod("__typeid__")
  ;
  ConverterMap::instance().addConverter(boost::make_shared<ConverterBoostDefSharedPtr<Psana::OceanOptics::DataV1> >(Pds::TypeId::Id_OceanOpticsData, 1));

  {
    PyObject* unvlist = PyList_New(1);
    PyList_SET_ITEM(unvlist, 0, PyObject_GetAttrString(submodule, "DataV1"));
    PyObject_SetAttrString(submodule, "Data", unvlist);
    Py_CLEAR(unvlist);
  }
  {
    PyObject* unvlist = PyList_New(1);
    PyList_SET_ITEM(unvlist, 0, PyObject_GetAttrString(submodule, "ConfigV1"));
    PyObject_SetAttrString(submodule, "Config", unvlist);
    Py_CLEAR(unvlist);
  }
  detail::register_ndarray_to_numpy_cvt<const uint16_t, 1>();
  detail::register_ndarray_to_numpy_cvt<const double, 1>();

} // createWrappers()
} // namespace OceanOptics
} // namespace psddl_python
