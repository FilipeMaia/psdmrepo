#define AAA 1

#include <psddl_pypsana/PyWrapper.h>
#include "MsgLogger/MsgLogger.h"
#include "PSEnv/Env.h"
#include "PSEvt/Event.h"
#include "PSEvt/EventId.h"
#include "ConfigSvc/ConfigSvc.h"
#include "psddl_pypsana/PyWrapper.h"
#include "python/Python.h"
#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/def.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/utility.hpp>
#include <numpy/arrayobject.h>
#include <string>

#include "psddl_psana/acqiris.ddl.h"
#include "psddl_pypsana/acqiris.ddl.wrapper.h"

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
using std::list;

using PSEnv::Env;
using PSEnv::EnvObjectStore;
using PSEvt::Event;
using PSEvt::EventKey;
using PSEvt::Source;
using Pds::Src;

typedef boost::shared_ptr<PyObject> PyObjPtr;

namespace Psana {

  namespace CreateWrappers {
    extern void createWrappers();
  }

  struct PyRefDelete {
    void operator()(PyObject* obj) { Py_CLEAR(obj); }
  };

  static boost::shared_ptr<char> none((char *)0);
  static const string NO_DEFAULT = "";
  static object Event_Class;
  static object EnvWrapper_Class;

  template<class T>
  string getTypeNameWithHighestVersion(std::map<string, T*> map, string typeNameGeneric) {
    string typeName = "";
    char v[256];
    for (int version = 1; true; version++) {
      sprintf(v, "V%d", version);
      string test = typeNameGeneric + v;
      if (! map.count(test)) {
        return typeName;
      }
      typeName = test;
    }
  }

  // Need wrapper because EnvObjectStore is boost::noncopyable
  class EnvObjectStoreWrapper {
  private:
    EnvObjectStore& _store;
  public:
    EnvObjectStoreWrapper(EnvObjectStore& store) : _store(store) {}
    // template <typename T> void putProxy(const boost::shared_ptr<PSEvt::Proxy<T> >& proxy, const Pds::Src& source);
    // template <typename T> void put(const boost::shared_ptr<T>& data, const Pds::Src& source);
    EnvObjectStore::GetResultProxy get1(const Src& src) { return _store.get(src); }
    EnvObjectStore::GetResultProxy get2(const Source& source, Src* foundSrc = 0) { return _store.get(source, foundSrc); }

    object get(const string& typeNameGeneric, const Source& source) {
      string typeName = getTypeNameWithHighestVersion<EnvGetter>(environmentGetter_map, typeNameGeneric);
      if (typeName == "") {
        return object(none);
      }
      EnvGetter *getter = environmentGetter_map[typeName];
      if (getter) {
        return getter->get(_store, source);
      }
      return object(none);
    }

    std::list<EventKey> keys(const Source& source = Source()) const { return _store.keys(); }
  };

  class EnvWrapper {
  private:
    Env& _env;
    const string& _name;
    const string& _className;
  public:
    EnvWrapper(Env& env, const string& name, const string& className) : _env(env), _name(name), _className(className) {}
    const string& jobName() const { return _env.jobName(); }
    const string& instrument() const { return _env.instrument(); }
    const string& experiment() const { return _env.experiment(); }
    const unsigned expNum() const { return _env.expNum(); }
    const string& calibDir() const { return _env.calibDir(); }
    EnvObjectStoreWrapper configStore() { return EnvObjectStoreWrapper(_env.configStore()); }
    PSEnv::EnvObjectStore& calibStore() { return _env.calibStore(); }
    PSEnv::EpicsStore& epicsStore() { return _env.epicsStore(); }
    RootHistoManager::RootHMgr& rhmgr() { return _env.rhmgr(); }
    PSHist::HManager& hmgr() { return _env.hmgr(); }
    Source configStr(const string& parameter, const string& _default = NO_DEFAULT) {
      ConfigSvc::ConfigSvc cfg;
      try {
        return cfg.getStr(_name, parameter);
      } catch (const ConfigSvc::ExceptionMissing& ex) {
        if (&_default == &NO_DEFAULT) {
          return cfg.getStr(_className, parameter);
        } else {
          return cfg.getStr(_className, parameter, _default);
        }
      }
    }
    Env& getEnv() { return _env; };
  };

  object getEnvWrapper(Env& env, const string& name, const string& className) {
    EnvWrapper _envWrapper(env, name, className);
    object envWrapper(EnvWrapper_Class(_envWrapper));
    return envWrapper;
  }

  object getEvtWrapper(Event& evt) {
    return object(Event_Class(evt));
  }

  object getFromEvent(Event& evt, const string& typeNameGeneric, Source& src) {
    string typeName = getTypeNameWithHighestVersion<EvtGetter>(eventGetter_map, typeNameGeneric);
    if (typeName == "") {
      return object(none);
    }
    EvtGetter *g = eventGetter_map[typeName];
    return g->get(evt, src);
  }

  void createWrappers()
  {
    // Required initialization of numpy array support
    _import_array();
    array::set_module_and_type("numpy", "ndarray");

#if 0

#define std_vector_class_(T) class_<vector<T> >("std_vector_" #T, no_init)\
    .def(vector_indexing_suite<vector<T> >())\
    .def("size", &vector<T>::size)

      class_<std::vector<X> >("XVec")
        .def(vector_indexing_suite<std::vector<X> >())
        ;

    //    std_vector_class_(FOO*);
#undef class_std_vector
#endif

    std_vector_class_(int);
    std_vector_class_(short);
    std_vector_class_(unsigned);
    std_vector_class_(unsigned short);

    class_<EnvObjectStore::GetResultProxy>("PSEnv::EnvObjectStore::GetResultProxy", no_init)
      ;

    class_<EnvObjectStoreWrapper>("PSEnv::EnvObjectStore", init<EnvObjectStore&>())
      .def("get1", &EnvObjectStoreWrapper::get1)
      .def("get2", &EnvObjectStoreWrapper::get2)
      .def("get", &EnvObjectStoreWrapper::get)
      .def("keys", &EnvObjectStoreWrapper::keys)
      ;

    class_<Source>("PSEvt::Source", no_init)
      .def("match", &Source::match)
      .def("isNoSource", &Source::isNoSource)
      .def("isExact", &Source::isExact)
      .def("src", &Source::src, return_value_policy<reference_existing_object>())
      ;

    Event_Class =
      class_<Event>("PSEvt::Event", init<Event&>())
      .def("get", &getFromEvent)
      ;

    EnvWrapper_Class =
      class_<EnvWrapper>("PSEnv::Env", init<EnvWrapper&>())
      .def("jobName", &EnvWrapper::jobName, return_value_policy<copy_const_reference>())
      .def("instrument", &EnvWrapper::instrument, return_value_policy<copy_const_reference>())
      .def("experiment", &EnvWrapper::experiment, return_value_policy<copy_const_reference>())
      .def("expNum", &EnvWrapper::expNum)
      .def("calibDir", &EnvWrapper::calibDir, return_value_policy<copy_const_reference>())
      .def("configStore", &EnvWrapper::configStore)
      .def("calibStore", &EnvWrapper::calibStore, return_value_policy<reference_existing_object>())
      .def("epicsStore", &EnvWrapper::epicsStore, return_value_policy<reference_existing_object>())
      .def("rhmgr", &EnvWrapper::rhmgr, return_value_policy<reference_existing_object>())
      .def("hmgr", &EnvWrapper::hmgr, return_value_policy<reference_existing_object>())
      .def("configStr", &EnvWrapper::configStr)
      ;

    CreateWrappers::createWrappers();
  }

  map<string, EvtGetter*> eventGetter_map;
  map<string, EnvGetter*> environmentGetter_map;

  PyObject* ndConvert(void* data, const unsigned* shape, const unsigned ndim, char *_ctype, size_t eltsize) {
    _import_array(); // XXX already done above
    array::set_module_and_type("numpy", "ndarray"); // XXX already done above

    npy_intp dims[ndim];
    int count = 1;
    for (unsigned int i = 0; i < ndim; i++) {
      count *= (dims[i] = shape[i]);
    }

    PyArray_Descr* descr = NULL;
    string ctype(_ctype);
    if (ctype == "int8_t") {
      descr = PyArray_DescrFromType(PyArray_BYTE);
    } else if (ctype == "uint8_t") {
      descr = PyArray_DescrFromType(PyArray_UBYTE);
    } else if (ctype == "int16_t") {
      descr = PyArray_DescrFromType(PyArray_SHORT);
    } else if (ctype == "uint16_t") {
      descr = PyArray_DescrFromType(PyArray_USHORT);
    } else if (ctype == "int32_t") {
      descr = PyArray_DescrFromType(PyArray_INT);
    } else if (ctype == "uint32_t") {
      descr = PyArray_DescrFromType(PyArray_UINT);
    } else if (ctype == "float") {
      descr = PyArray_DescrFromType(PyArray_CFLOAT);
    } else if (ctype == "double") {
      descr = PyArray_DescrFromType(PyArray_CDOUBLE);
    } else {
      printf("!!!!!!!!! ndConvert(%s):%d\n", _ctype, __LINE__);
      printf("!!!!!!!!! ndConvert(%s):%d: pointer=%p\n", _ctype, __LINE__, data);
      printf("!!!!!!!!! ndConvert(%s):%d: data=0x%x\n", _ctype, __LINE__, *(char *)data);
      printf("!!!!!!!!! ndConvert(%s):%d: eltsize=%d\n", _ctype, __LINE__, (int) eltsize);
      printf("Cannot convert ctype %s to PyArray type; will approximate with bytes\n", ctype.c_str());

      descr = PyArray_DescrFromType(PyArray_BYTE);
    }
    PyObject* result = PyArray_NewFromDescr(&PyArray_Type, descr, ndim, dims, NULL, data, 0, NULL);
      //PyObject* result = PyArray_SimpleNewFromData(ndim, dims, ptype, data);
    return result;
  }

  // call specific method
  boost::shared_ptr<PyObject> call(PyObject* method, Event& evt, Env& env, const string& name, const string& className)
  {
    object envWrapper = Psana::getEnvWrapper(env, name, className);
    object evtWrapper = Psana::getEvtWrapper(evt);

    PyObjPtr args(PyTuple_New(2), PyRefDelete());
    PyTuple_SET_ITEM(args.get(), 0, evtWrapper.ptr());
    PyTuple_SET_ITEM(args.get(), 1, envWrapper.ptr());
    PyObjPtr res(PyObject_Call(method, args.get(), NULL), PyRefDelete());
    return res;
  }


} // namespace Psana
