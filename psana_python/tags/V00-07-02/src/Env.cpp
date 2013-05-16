//--------------------------------------------------------------------------
// File and Version Information:
//  $Id: Env.cpp 4455 2012-09-12 00:22:58Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//  Class Env...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "psana_python/Env.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/python.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psana_python/EnvObjectStore.h"
#include "psana_python/EpicsStore.h"
#include "pytools/make_pyshared.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using namespace psana_python;

namespace {

  // type-specific methods
  PyObject* Env_fwkName(PyObject* self, PyObject*);
  PyObject* Env_jobName(PyObject* self, PyObject*);
  PyObject* Env_instrument(PyObject* self, PyObject*);
  PyObject* Env_experiment(PyObject* self, PyObject*);
  PyObject* Env_expNum(PyObject* self, PyObject*);
  PyObject* Env_calibDir(PyObject* self, PyObject*);
  PyObject* Env_configStore(PyObject* self, PyObject*);
  PyObject* Env_calibStore(PyObject* self, PyObject*);
  PyObject* Env_epicsStore(PyObject* self, PyObject*);
  PyObject* Env_hmgr(PyObject* self, PyObject*);
  PyObject* Env_subprocess(PyObject* self, PyObject*);
  PyObject* Env_getConfig(PyObject* self, PyObject* args);

  PyMethodDef methods[] = {
    { "fwkName",       Env_fwkName,       METH_NOARGS, 
        "self.fwkName() -> str\n\nReturns name of the framework. This method is supposed to be defined across different frameworks. "
        "It returns the name of the current framework, e.g. when client code runs inside pyana framework it will return string "
        "\"pyana\", inside  psana framework it will return \"psana\". This method should be used as a primary mechanism for " 
        "distinguishing between different frameworks in cases when client needs to execute framework-specific code."},
    { "jobName",       Env_jobName,       METH_NOARGS, "self.jobName() -> str\n\nReturns job name."},
    { "instrument",    Env_instrument,    METH_NOARGS, "self.instrument() -> str\n\nReturns instrument name."},
    { "experiment",    Env_experiment,    METH_NOARGS, "self.experiment() -> str\n\nReturns experiment name."},
    { "expNum",        Env_expNum,        METH_NOARGS, "self.expNum() -> int\n\nReturns experiment number or 0."},
    { "calibDir",      Env_calibDir,      METH_NOARGS, 
        "self.calibDir() -> str\n\nReturns path the calibration directory for current instrument/experiment, "
        "typically \"/reg/d/psdm/INSTR/exper/calib\" but can be changed from in job configuration."},
    { "configStore",   Env_configStore,   METH_NOARGS,
        "self.configStore() -> object\n\nAccess to Configuration Store (:py:class:`EnvObjectStore`) object."},
    { "calibStore",    Env_calibStore,    METH_NOARGS,
        "self.calibStore() -> object\n\nAccess to Calibration Store (:py:class:`EnvObjectStore`) object."},
    { "epicsStore",    Env_epicsStore,    METH_NOARGS, "self.epicsStore() -> object\n\nAccess to EPICS Store (:py:class:`EpicsStore`) object."},
    { "hmgr",          Env_hmgr,          METH_NOARGS, "self.hmgr() -> object\n\nAccess to histogram manager."},
    { "subprocess",    Env_subprocess,    METH_NOARGS,
        "self.subprocess() -> int\n\nReturns subprocess number or 0 if running inside main process. If multi-processssing "
        "is disabled always returns 0. Currently psana does not support multi-processing."},
    { "getConfig",     Env_getConfig,     METH_VARARGS, 
        "self.getConfig(...) -> object\n\nPyana compatibility method, shortcut for ``self.configStore().get()``, deprecated."},
    {0, 0, 0, 0}
   };

  char typedoc[] = "\
Python wrapper for psana Env C++ class. Instances of this class are \
created by the framework itself and returned to the client from framework methods.\
";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
psana_python::Env::initType(PyObject* module)
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType("Env", module, "psana");
}

// Dump object info to a stream
void 
psana_python::Env::print(std::ostream& out) const
{
  out << "psana.Env()" ;
}

namespace {

PyObject*
Env_fwkName(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  return PyString_FromString(cself->fwkName().c_str());
}

PyObject*
Env_jobName(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  return PyString_FromString(cself->jobName().c_str());
}

PyObject*
Env_instrument(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  return PyString_FromString(cself->instrument().c_str());
}

PyObject*
Env_experiment(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  return PyString_FromString(cself->experiment().c_str());
}

PyObject*
Env_expNum(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  return PyInt_FromLong(cself->expNum());
}

PyObject*
Env_calibDir(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  return PyString_FromString(cself->calibDir().c_str());
}

PyObject*
Env_configStore(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  return EnvObjectStore::PyObject_FromCpp(cself->configStore().shared_from_this());
}

PyObject*
Env_calibStore(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  return EnvObjectStore::PyObject_FromCpp(cself->calibStore().shared_from_this());
}

PyObject*
Env_epicsStore(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  return EpicsStore::PyObject_FromCpp(cself->epicsStore().shared_from_this());
}

PyObject*
Env_hmgr(PyObject* self, PyObject*)
{
  boost::shared_ptr<PSEnv::Env>& cself = Env::cppObject(self);
  boost::python::object boo(cself->hmgr());
  Py_INCREF(boo.ptr());
  return boo.ptr();
}

PyObject*
Env_subprocess(PyObject* self, PyObject*)
{
  return PyInt_FromLong(0);
}

PyObject* 
Env_getConfig(PyObject* self, PyObject* args)
{
  // forward call to the config store method
  pytools::pyshared_ptr cfgStore = pytools::make_pyshared(Env_configStore(self, 0));
  pytools::pyshared_ptr method = pytools::make_pyshared(PyObject_GetAttrString(cfgStore.get(), "get"));
  return PyObject_CallObject(method.get(), args);
}

}