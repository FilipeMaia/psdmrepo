//--------------------------------------------------------------------------
// File and Version Information:
//  $Id$
//
// Description:
//  Class Run...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Run.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "EventIter.h"
#include "ScanIter.h"
#include "psana_python/EnvWrapper.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // type-specific methods
  PyObject* Run_scans(PyObject* self, PyObject*);
  PyObject* Run_events(PyObject* self, PyObject*);
  PyObject* Run_nonzero(PyObject* self, PyObject*);
  PyObject* Run_env(PyObject* self, PyObject*);

  PyMethodDef methods[] = {
    { "scans",       Run_scans,     METH_NOARGS, "self.scans() -> iterator\n\nReturns iterator for contained scans (:py:class:`ScanIter`)" },
    { "events",      Run_events,    METH_NOARGS, "self.events() -> iterator\n\nReturns iterator for contained events (:py:class:`EventIter`)" },
    { "env",         Run_env,       METH_NOARGS, "self.env() -> object\n\nReturns environment object" },
    { "__nonzero__", Run_nonzero,   METH_NOARGS, "self.__nonzero__() -> bool\n\nReturns true for non-null object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python wrapper for psana Run type. Run type represents data originating "
      "from a single run and it contains one or more scans (calib cycles) which in turn contain "
      "events. This class provides ways to iterate over individual scans in a run or over all "
      "events contained in all scans of this run. Actual iteration is implemented in "
      ":py:class:`ScanIter` and :py:class:`EventIter` classes, this class serves as a factory "
      "for iterator instances.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
psana_python::pyext::Run::initType(PyObject* module)
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType("Run", module);
}

namespace {

PyObject*
Run_scans(PyObject* self, PyObject* )
{
  psana_python::pyext::Run* py_this = static_cast<psana_python::pyext::Run*>(self);
  return psana_python::pyext::ScanIter::PyObject_FromCpp(py_this->m_obj.scans());
}

PyObject*
Run_events(PyObject* self, PyObject* )
{
  psana_python::pyext::Run* py_this = static_cast<psana_python::pyext::Run*>(self);
  return psana_python::pyext::EventIter::PyObject_FromCpp(py_this->m_obj.events());
}

PyObject*
Run_env(PyObject* self, PyObject* )
{
  psana_python::pyext::Run* py_this = static_cast<psana_python::pyext::Run*>(self);
  PSEnv::Env& env = py_this->m_obj.env();
  boost::python::object envWrapper(psana_python::EnvWrapper(env.shared_from_this(), "", ""));
  PyObject* envObj = envWrapper.ptr();
  Py_INCREF(envObj);
  return envObj;
}

PyObject*
Run_nonzero(PyObject* self, PyObject* )
{
  psana_python::pyext::Run* py_this = static_cast<psana_python::pyext::Run*>(self);
  return PyBool_FromLong(long(bool(py_this->m_obj)));
}

}
