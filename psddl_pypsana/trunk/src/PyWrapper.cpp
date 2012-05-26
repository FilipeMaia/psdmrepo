#define AAA 1

#include <psddl_pypsana/PyWrapper.h>
#include "MsgLogger/MsgLogger.h"
#include "PSEnv/Env.h"
#include "PSEvt/Event.h"
#include "PSEvt/EventId.h"
#include "PSEvt/ProxyDictI.h"
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
#include <set>

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
using std::set;
using std::string;
using std::vector;
using std::list;

using PSEnv::Env;
using PSEnv::EnvObjectStore;
using PSEnv::EpicsStore;
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
  static object Event_Class;
  static object EnvWrapper_Class;
  static map<string, EvtGetter*> evtGetter_map;
  static map<string, EnvGetter*> envGetter_map;

  template<class T>
  string getTypeNameWithHighestVersion(map<string, T*> map, string typeNameGeneric) {
    if (map.count(typeNameGeneric)) {
      return typeNameGeneric;
    }
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
    EnvObjectStore::GetResultProxy getBySrc(const Src& src) { return _store.get(src); }
    EnvObjectStore::GetResultProxy getBySource(const Source& source, Src* foundSrc = 0) { return _store.get(source, foundSrc); }

    object getByType(const string& typeNameGeneric, const Source& source) {
      string typeName = getTypeNameWithHighestVersion<EnvGetter>(envGetter_map, typeNameGeneric);
      if (typeName == "") {
        return object(none);
      }
      EnvGetter *getter = envGetter_map[typeName];
      if (getter) {
        return getter->get(_store, source);
      }
      return object(none);
    }

    list<EventKey> keys(const Source& source = Source()) const { return _store.keys(); }
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
    Env& getEnv() { return _env; };

    void printAllKeys() {
      EnvObjectStore& store = _env.configStore();
      list<EventKey> keys = store.keys(); // , Source());
      list<EventKey>::iterator it;
      for (it = keys.begin(); it != keys.end(); it++) {
        cout << "THIS is an ENV key: " << *it << endl;
      }
    }

    void printConfigKeys() {
      ConfigSvc::ConfigSvc cfg;
      list<string> keys = cfg.getKeys(_name);
      list<string>::iterator it;
      cout << "!!! keys.size() = " << keys.size() << " for " << _name << endl;
      for (it = keys.begin(); it != keys.end(); it++) {
        cout << "THIS is an ConfigSvc key: " << *it << endl;
      }
    }

    const char* configStr(const string& parameter) {
      ConfigSvc::ConfigSvc cfg;
      try {
        return cfg.getStr(_name, parameter).c_str();
      } catch (const ConfigSvc::ExceptionMissing& ex) {
        try {
          return cfg.getStr(_className, parameter).c_str();
        } catch (const ConfigSvc::ExceptionMissing& ex) {
          return 0;
        }
      }
    }

    string configStr2(const string& parameter, const char* _default) {
      if (_default == 0) {
        return configStr(parameter);
      }
      ConfigSvc::ConfigSvc cfg;
      try {
        return cfg.getStr(_name, parameter);
      } catch (const ConfigSvc::ExceptionMissing& ex) {
        return cfg.getStr(_className, parameter, _default);
      }
    }

    Source configSource(const string& _default) {
      const char* value = configStr("source");
      if (value) {
        return Source(value);
      } else {
        return Source(_default);
      }
    }

    object getConfigByType(const char* typeNameGeneric, const char* detectorSourceName) {
      const Source detectorSource(detectorSourceName);
      string typeName = getTypeNameWithHighestVersion<EnvGetter>(envGetter_map, typeNameGeneric);
      if (typeName == "") {
        return object(none);
      }
      EnvGetter *getter = envGetter_map[typeName];
      if (getter) {
        return getter->get(_env.configStore(), detectorSource);
      }
      return object(none);
    }

    object getConfig(int typeId, const char* detectorSourceName) {
      char name[64];
      sprintf(name, "@EnvType_%d_", typeId);
      return getConfigByType(name, detectorSourceName);
    }
  };

  object getEnvWrapper(Env& env, const string& name, const string& className) {
    EnvWrapper _envWrapper(env, name, className);
    object envWrapper(EnvWrapper_Class(_envWrapper));
    return envWrapper;
  }

  object getEvtWrapper(Event& evt) {
    return object(Event_Class(evt));
  }

  boost::shared_ptr<string> get_Event(Event& evt, const string& key) {
    return boost::shared_ptr<string>(evt.get(key));
  }

  void printAllKeys_Event(Event& evt) {
    Event::GetResultProxy proxy = evt.get();
    list<EventKey> keys;
    proxy.m_dict->keys(keys, Source());
    list<EventKey>::iterator it;
    for (it = keys.begin(); it != keys.end(); it++) {
      cout << "THIS is a key: " << *it << endl;
    }
  }

  int run_Event(Event& evt) {
    const boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
    return eventId->run();
  }

#if 0
  object getBySourceAndKey_Event(Event& evt, const string& typeNameGeneric, Source& source, const string& key) {
    string typeName = getTypeNameWithHighestVersion<EvtGetter>(evtGetter_map, typeNameGeneric);
    if (typeName == "") {
      return object(none);
    }
    EvtGetter *g = evtGetter_map[typeName];
    return g->get(evt, source, key);
  }
#endif

  object getByType_Event(Event& evt, const string& typeNameGeneric, const string& detectorSourceName) {
    if (typeNameGeneric == "PSEvt::EventId") {
      const boost::shared_ptr<PSEvt::EventId> eventId = evt.get();
      return object(eventId);
    }

    printAllKeys_Event(evt);
    Source detectorSource;
    if (detectorSourceName == "") {
      detectorSource = Source();
    } else {
      detectorSource = Source(detectorSourceName);
    }
    string typeName = getTypeNameWithHighestVersion<EvtGetter>(evtGetter_map, typeNameGeneric);
    if (typeName == "") {
      return object(none);
    }
    EvtGetter *g = evtGetter_map[typeName];
    return g->get(evt, detectorSource, string());
  }

  object getByType1_Event(Event& evt, const string& typeName) {
    printAllKeys_Event(evt);

    //Event::GetResultProxy proxy = evt.get();
    //boost::shared_ptr<ProxyDictI> dict = proxy.m_dict;

    //boost::shared_ptr<void> vptr = m_dict->get(&typeid(const T), m_source, m_key, m_foundSrc);
    //return boost::static_pointer_cast<T>(vptr);

    return object(none); // @@@
  }

#if 0
  // Src is a module type aka fullName
  object getBySrcAndKey_Event(Event& evt, const string& typeNameGeneric, Src& src, const string& key) {
    string typeName = getTypeNameWithHighestVersion<EvtGetter>(evtGetter_map, typeNameGeneric);
    if (typeName == "") {
      return object(none);
    }

    // @@@
    // bool exists = evt.exists

    EvtGetter *g = evtGetter_map[typeName];
    return g->get(evt, src, key);
  }

  // Src is a module type aka fullName
  object getBySrc_Event(Event& evt, const string& typeNameGeneric, Src& src) {
    return getBySrcAndKey_Event(evt, typeNameGeneric, src, string());
  }
#endif

  PyObject* ndConvert(const unsigned ndim, const unsigned* shape, int ptype, void* data) {
    npy_intp dims[ndim];
    for (unsigned i = 0; i < ndim; i++) {
      dims[i] = shape[i];
    }
    return PyArray_SimpleNewFromData(ndim, dims, ptype, data);
  }

  static bool createWrappersDone = false;

  void createWrappers()
  {
    if (createWrappersDone) {
      return;
    }

    // Required initialization of numpy array support
    _import_array();
    array::set_module_and_type("numpy", "ndarray");

    std_vector_class_(int);
    std_vector_class_(short);
    std_vector_class_(unsigned);
    std_vector_class_(unsigned short);

    class_<EnvObjectStore::GetResultProxy>("PSEnv::EnvObjectStore::GetResultProxy", no_init)
      ;

    class_<EnvObjectStoreWrapper>("PSEnv::EnvObjectStore", init<EnvObjectStore&>())
      .def("getBySrc", &EnvObjectStoreWrapper::getBySrc)
      .def("getBySource", &EnvObjectStoreWrapper::getBySource)
      .def("getByType", &EnvObjectStoreWrapper::getByType)
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
      .def("get", &get_Event)
      .def("getByType", &getByType_Event)
      .def("getByType1", &getByType1_Event)
      //.def("getBySourceAndKey", &getBySourceAndKey_Event)
      //.def("getBySrcAndKey", &getBySrcAndKey_Event)
      //.def("getBySrc", &getBySrc_Event)
      .def("printAllKeys", &printAllKeys_Event)
      .def("run", &run_Event)
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
      .def("configSource", &EnvWrapper::configSource)
      .def("configStr", &EnvWrapper::configStr)
      .def("configStr2", &EnvWrapper::configStr2)
      .def("printAllKeys", &EnvWrapper::printAllKeys)
      .def("printConfigKeys", &EnvWrapper::printConfigKeys)
      .def("getConfigByType", &EnvWrapper::getConfigByType)
      .def("getConfig", &EnvWrapper::getConfig)
      ;

    CreateWrappers::createWrappers();

    createWrappersDone = true;
  }

  void addEnvGetter(EnvGetter* getter) {
    envGetter_map[getter->getTypeName()] = getter;
    printf("~~~ adding %s\n", getter->getTypeName());
    if (getter->getTypeId() != -1) {
      char name[64];
      sprintf(name, "@EnvType_%d_V%d", getter->getTypeId(), getter->getVersion());
      envGetter_map[name] = getter;
      printf("~~~ adding %s\n", name);
    }
  }

  void addEvtGetter(EvtGetter* getter) {
    evtGetter_map[getter->getTypeName()] = getter;
    printf("~~~ adding %s\n", getter->getTypeName());
    if (getter->getTypeId() != -1) {
      char name[64];
      sprintf(name, "@EvtType_%d_V%d", getter->getTypeId(), getter->getVersion());
      evtGetter_map[name] = getter;
      printf("~~~ adding %s\n", name);
    }
  }

  static set<string> class_set;

  bool class_needed(const char* _ctype) {
#if 1
    return true;
#else
    const string ctype(_ctype);
    if (class_set.count(ctype) == 0) {
      class_set.insert(ctype); // assume that it will be added by the caller
      return true;
    }
    return false;
#endif
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
