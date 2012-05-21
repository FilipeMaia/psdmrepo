#ifndef PSANA_PYWRAPPER_H
#define PSANA_PYWRAPPER_H 1

#include <string>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>
#include "PSEnv/Env.h"
#include "PSEvt/Event.h"

namespace Psana {
  class EvtGetter {
  public:
    virtual std::string getTypeName() = 0;
    virtual boost::python::api::object get(PSEvt::Event& evt, PSEvt::Source& src) = 0;
    virtual ~EvtGetter() {}
  };

  class EnvGetter {
  public:
    virtual std::string getTypeName() = 0;
    virtual boost::python::api::object get(PSEnv::EnvObjectStore& store, const PSEvt::Source& src) = 0;
    virtual ~EnvGetter() {}
  };

  extern std::map<std::string, EvtGetter*> evtGetter_map;
  extern std::map<std::string, EnvGetter*> envGetter_map;

  extern PyObject* ndConvert(const unsigned ndim, const unsigned* shape, int ptype, void* data);
  extern void createWrappers();
  extern boost::shared_ptr<PyObject> call(PyObject* method, PSEvt::Event& evt, PSEnv::Env& env, const std::string& name, const std::string& className);
}

#define std_vector_class_(T) boost::python::class_<vector<T> >("std::vector<" #T ">")\
    .def(boost::python::vector_indexing_suite<std::vector<T> >())

#define associate_PyArrayType(ctype, ptype) const int PyArray_ ## ctype = ptype
associate_PyArrayType(int8_t, PyArray_BYTE);
associate_PyArrayType(uint8_t, PyArray_UBYTE);
associate_PyArrayType(int16_t, PyArray_SHORT);
associate_PyArrayType(uint16_t, PyArray_USHORT);
associate_PyArrayType(int32_t, PyArray_INT);
associate_PyArrayType(uint32_t, PyArray_UINT);
associate_PyArrayType(float, PyArray_CFLOAT);
associate_PyArrayType(double, PyArray_CDOUBLE);

#define ND_CONVERT(value, ctype, ndim) const ndarray<ctype, ndim>& a(value); return Psana::ndConvert(ndim, a.shape(), PyArray_ ## ctype, (void *) a.data())
#define VEC_CONVERT(value, ctype) const ndarray<ctype, 1>& a(value); const std::vector<ctype> v(a.data(), a.data() + a.size()); return v
#define EVT_GETTER(x) {Psana::EvtGetter *g = new x ## _EvtGetter(); Psana::evtGetter_map[g->getTypeName()] = g;}
#define ENV_GETTER(x) {Psana::EnvGetter *g = new x ## _EnvGetter(); Psana::envGetter_map[g->getTypeName()] = g;}

#endif // PSANA_PYWRAPPER_H
