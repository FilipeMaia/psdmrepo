#include <psddl_pypsana/PyWrapper.h>
#include "MsgLogger/MsgLogger.h"
#include "PSEnv/Env.h"
#include "PSEvt/Event.h"
#include "PSEvt/EventId.h"
#include "psddl_pypsana/PyWrapper.h"
#include "python/Python.h"
#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/def.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/utility.hpp>
#include <numpy/arrayobject.h>
#include <string>

using boost::python::api::object;
using boost::python::class_;
using boost::python::copy_const_reference;
using boost::python::init;
using boost::python::no_init;
using boost::python::numeric::array;
using boost::python::reference_existing_object;
using boost::python::return_value_policy;
using boost::python::vector_indexing_suite;

using std::map;
using std::string;
using std::vector;

using PSEnv::Env;
using PSEvt::Event;
using PSEvt::Source;

typedef boost::shared_ptr<PyObject> PyObjPtr;

namespace Psana {

  struct PyRefDelete {
    void operator()(PyObject* obj) { Py_CLEAR(obj); }
  };

  class EnvWrapper {
  private:
    Env& _env;
  public:
    EnvWrapper(Env& env) : _env(env) {}
    const string& jobName() const { return _env.jobName(); }
    const string& instrument() const { return _env.instrument(); }
    const string& experiment() const { return _env.experiment(); }
    const unsigned expNum() const { return _env.expNum(); }
    const string& calibDir() const { return _env.calibDir(); }
    PSEnv::EnvObjectStore& configStore() { return _env.configStore(); }
    PSEnv::EnvObjectStore& calibStore() { return _env.calibStore(); }
    PSEnv::EpicsStore& epicsStore() { return _env.epicsStore(); }
    RootHistoManager::RootHMgr& rhmgr() { return _env.rhmgr(); }
    PSHist::HManager& hmgr() { return _env.hmgr(); }
    Source configStr(const string& parameter, const string& typeName);
    Env& getEnv() { return _env; };
  };

  static object Event_Class;
  static object EnvWrapper_Class;

  object getEnvWrapper(Env& env) {
    EnvWrapper _envWrapper(env);
    object envWrapper(EnvWrapper_Class(_envWrapper));
    return envWrapper;
  }

  object getEvtWrapper(Event& evt) {
    return object(Event_Class(evt));
  }

  Source EnvWrapper::configStr(const string& parameter, const string& typeName) {
#if 0
    return ::configStr(parameter, typeName);
#else
    return typeName; // XXX
#endif
  }

  Source _configStr(const string& parameter, const string& typeName) {
    return typeName;
  }

  static boost::shared_ptr<char> none((char *)0);

  object getData(Event& evt, const string& typeName, Source& src) {
    if (eventGetter_map.count(typeName)) {
      EvtGetter *g = eventGetter_map[typeName];
      return g->get(evt, src);
    }
    return object(none);
  }

  object getConfig(EnvWrapper& env, const string& typeName, Source& src) {
    if (environmentGetter_map.count(typeName)) {
      EnvGetter *g = environmentGetter_map[typeName];
      return g->get(env.getEnv(), src);
    }
    return object(none);
  }

  namespace CreateWrappers {
    extern void createWrappers();
  }

  void createWrappers()
  {
    // Required initialization of numpy array support
    _import_array();
    array::set_module_and_type("numpy", "ndarray");

#define class_std_vector(T) class_<vector<T> >("std_vector_" #T, no_init)\
    .def(vector_indexing_suite<vector<T> >())\
    .def("size", &vector<T>::size)

    class_std_vector(int);
    class_std_vector(short);
    class_std_vector(unsigned);
    class_std_vector(unsigned short);
#undef class_std_vector

    class_<PSEvt::Source>("PSEvt::Source", no_init)
      .def("match", &PSEvt::Source::match)
      .def("isNoSource", &PSEvt::Source::isNoSource)
      .def("isExact", &PSEvt::Source::isExact)
      .def("src", &PSEvt::Source::src, return_value_policy<reference_existing_object>())
      ;

    Event_Class =
      class_<Event>("Event", init<Event&>())
      .def("getData", &getData) // TEMPORARY
      ;

    EnvWrapper_Class =
      class_<EnvWrapper>("Env", init<EnvWrapper&>())
      .def("jobName", &EnvWrapper::jobName, return_value_policy<copy_const_reference>())
      .def("instrument", &EnvWrapper::instrument, return_value_policy<copy_const_reference>())
      .def("experiment", &EnvWrapper::experiment, return_value_policy<copy_const_reference>())
      .def("expNum", &EnvWrapper::expNum)
      .def("calibDir", &EnvWrapper::calibDir, return_value_policy<copy_const_reference>())
      .def("configStore", &EnvWrapper::configStore, return_value_policy<reference_existing_object>())
      .def("calibStore", &EnvWrapper::calibStore, return_value_policy<reference_existing_object>())
      .def("epicsStore", &EnvWrapper::epicsStore, return_value_policy<reference_existing_object>())
      // rhmgr
      // hmgr
      .def("configStr", &EnvWrapper::configStr)
      .def("getConfig", &getConfig)
      ;

    CreateWrappers::createWrappers();
  }

  map<string, EvtGetter*> eventGetter_map;
  map<string, EnvGetter*> environmentGetter_map;

  PyObject* ndConvert(void* data, const unsigned* shape, const unsigned ndim, char *_ctype) {
    _import_array();
    array::set_module_and_type("numpy", "ndarray");

    string ctype(_ctype);
    int ptype;

    if (ctype.find("::") != string::npos) {
      ptype = PyArray_OBJECT;
    } else if (ctype == "int8_t") {
      ptype = PyArray_BYTE;
    } else if (ctype == "uint8_t") {
      ptype = PyArray_UBYTE;
    } else if (ctype == "int16_t") {
      ptype = PyArray_SHORT;
    } else if (ctype == "uint16_t") {
      ptype = PyArray_USHORT;
    } else if (ctype == "int32_t") {
      ptype = PyArray_INT;
    } else if (ctype == "uint32_t") {
      ptype = PyArray_UINT;
    } else if (ctype == "float") {
      ptype = PyArray_CFLOAT; // XXX ???
    } else if (ctype == "double") {
      ptype = PyArray_CDOUBLE; // XXX ???
    } else {
      printf("Cannot convert ctype %s to PyArray type\n", ctype.c_str());
      exit(1);
    }

    npy_intp dims[ndim];
    for (unsigned int i = 0; i < ndim; i++) {
      dims[i] = shape[i];
    }
    PyObject* result = PyArray_SimpleNewFromData(ndim, dims, ptype, data);
    return result;
  }

  // call specific method
  boost::shared_ptr<PyObject> call(PyObject* method, Event& evt, Env& env)
  {
    object envWrapper = Psana::getEnvWrapper(env);
    object evtWrapper = Psana::getEvtWrapper(evt);

    PyObjPtr args(PyTuple_New(2), PyRefDelete());
    PyTuple_SET_ITEM(args.get(), 0, evtWrapper.ptr());
    PyTuple_SET_ITEM(args.get(), 1, envWrapper.ptr());

    PyObjPtr res(PyObject_Call(method, args.get(), NULL), PyRefDelete());
    return res;
  }


} // namespace Psana
