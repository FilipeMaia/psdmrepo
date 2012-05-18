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

  extern PyObject* ndConvert(void* data, const unsigned* shape, const unsigned ndim, char* ctype, size_t eltsize);
  extern std::map<std::string, EvtGetter*> eventGetter_map;
  extern std::map<std::string, EnvGetter*> environmentGetter_map;
  extern void createWrappers();
  extern boost::shared_ptr<PyObject> call(PyObject* method, PSEvt::Event& evt, PSEnv::Env& env, const std::string& name, const std::string& className);
}

#define ND_CONVERT(value, ctype, ndim) const ndarray<ctype, ndim>& a(value); return Psana::ndConvert((void *) a.data(), a.shape(), ndim, #ctype, sizeof(ctype))
#define VEC_CONVERT(value, ctype) const ndarray<ctype, 1>& a(value); const std::vector<ctype> v(a.data(), a.data() + a.size()); return v
#define EVT_GETTER(x) {Psana::EvtGetter *g = new x ## _EvtGetter(); Psana::eventGetter_map[g->getTypeName()] = g;}
#define ENV_GETTER(x) {Psana::EnvGetter *g = new x ## _EnvGetter(); Psana::environmentGetter_map[g->getTypeName()] = g;}
#define std_vector_class_(T) boost::python::class_<vector<T> >("std::vector<" #T ">")\
    .def(boost::python::vector_indexing_suite<std::vector<T> >())

#endif // PSANA_PYWRAPPER_H
