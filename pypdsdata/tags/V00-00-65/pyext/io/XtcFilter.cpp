//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFilter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcFilter.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../types/TypeLib.h"
#include "../Dgram.h"
#include "../Xtc.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // destructor method for data buffers allocated here
  void _destroyXtc(Pds::Xtc* obj) {
    delete [] ((char*)obj);
  }
  void _destroyDgram(Pds::Dgram* obj) {
    delete [] ((char*)obj);
  }

  // standard Python stuff
  int XtcFilter_init(PyObject* self, PyObject* args, PyObject* kwds);

  // type-specific methods
  PyObject* XtcFilter_filter(PyObject* self, PyObject* args);

  PyMethodDef methods[] = {
    { "filter",    XtcFilter_filter,      METH_VARARGS,
        "self.filter(object) -> object\n\n"
        "This method does actual filtering job. Note also that for some types\n"
        "of damage it may need to skip damaged data if the structure  of XTC\n"
        "cannot be recovered. This happens independently of content-based\n"
        "filtering. Method accepts object which must be an instance of type\n"
        "xtc.Dgram or xtc.Xtc. It returns object of the same type with some\n"
        "contents removed. It can also return None if all contents was removed\n"
        "depending on constructor parameters values." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ XtcInput::XtcFilter class.\n\n"
      "Constructor takes one required argument (callable) which will be called with\n"
      "xtc.Xtc object as parameter and should return True or False. Keyword\n"
      "arguments to constructor are *keepEmptyCont* (def: False), *keepEmptyDgram*\n"
      "(def: False), and *keepAny* (def: False). These are the same arguments as\n"
      "passed to C++ instance constructor.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::XtcFilter::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_init = XtcFilter_init;

  BaseType::initType( "XtcFilter", module );
}

pypdsdata::XtcFilter_CallableWrapper::XtcFilter_CallableWrapper(PyObject* obj)
  : m_obj(obj)
{
  Py_INCREF(m_obj);
}

pypdsdata::XtcFilter_CallableWrapper::XtcFilter_CallableWrapper(const XtcFilter_CallableWrapper& other)
  : m_obj(other.m_obj)
{
  Py_INCREF(m_obj);
}

pypdsdata::XtcFilter_CallableWrapper&
pypdsdata::XtcFilter_CallableWrapper::operator=(const XtcFilter_CallableWrapper& other)
{
  if (this != &other) {
    Py_CLEAR(m_obj);
    m_obj = other.m_obj;
    Py_INCREF(m_obj);
  }
  return *this;
}

pypdsdata::XtcFilter_CallableWrapper::~XtcFilter_CallableWrapper()
{
  Py_CLEAR(m_obj);
}

bool
pypdsdata::XtcFilter_CallableWrapper::operator()(const Pds::Xtc* input) const
{

  // build argument list
  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pypdsdata::Xtc::PyObject_FromPds(const_cast<Pds::Xtc*>(input), 0, input->extent));

  // call our guy
  PyObject* ret = PyObject_Call(m_obj, args, 0);

  // convert result to bool
  bool result = false;

  // we can handle integer (bool is int) or None
  if (ret == Py_None) {
    // OK
  } else if (PyInt_Check(ret)) {
    result = bool(PyInt_AS_LONG(ret));
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Error: callable object returned non-integer");
  }

  // cleanup
  Py_CLEAR(ret);
  Py_CLEAR(args);

  return result;
}


namespace {

int
XtcFilter_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::XtcFilter* py_this = (pypdsdata::XtcFilter*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  static char *kwlist[] = {"callable", "keepEmptyCont", "keepEmptyDgram", "keepAny", 0};
  PyObject* obj = 0;
  int keepEmptyCont = 0;
  int keepEmptyDgram = 0;
  int keepAny = 0;
  if (not PyArg_ParseTupleAndKeywords(args, kwds, "O|iii:XtcFilter", kwlist,
      &obj, &keepEmptyCont, &keepEmptyDgram, &keepAny)) return -1;

  // check that object is callable
  if (not PyCallable_Check(obj)) {
    PyErr_SetString(PyExc_ValueError, "Error: non-callable object given to XtcFilter");
    return -1;
  }

  pypdsdata::XtcFilter_CallableWrapper callable(obj);
  new (&py_this->m_obj) pypdsdata::XtcFilter::PdsType(callable, keepEmptyCont, keepEmptyDgram, keepAny);

  return 0;
}

PyObject*
XtcFilter_filter(PyObject* self, PyObject* args)
{
  pypdsdata::XtcFilter::PdsType& cpp_self = pypdsdata::XtcFilter::pdsObject(self);

  // get arguments
  PyObject* obj = 0;
  if (not PyArg_ParseTuple(args, "O:XtcFilterTypeId", &obj)) return 0;

  // may be Xtc of Dgram
  if (pypdsdata::Xtc::Object_TypeCheck(obj)) {

    // get XTC
    const Pds::Xtc* xtc = pypdsdata::Xtc::pdsObject(obj);

    // allocate buffer
    char* buf = new char[xtc->extent];

    // call C++ code
    size_t newSize = cpp_self.filter(xtc, buf);

    if (newSize) {

      // make new xtc object
      Pds::Xtc* newXtc = (Pds::Xtc*)buf;
      return pypdsdata::Xtc::PyObject_FromPds(newXtc, 0, newSize, ::_destroyXtc);

    } else {

      Py_RETURN_NONE;

    }


  } else if (pypdsdata::Dgram::Object_TypeCheck(obj)) {

    // get datagram
    const Pds::Dgram* dg = pypdsdata::Dgram::pdsObject(obj);

    // allocate buffer
    char* buf = new char[sizeof(Pds::Dgram) + dg->xtc.sizeofPayload()];

    // call C++ code
    size_t newSize = cpp_self.filter(dg, buf);

    if (newSize) {

      // make new xtc object
      Pds::Dgram* newDg = (Pds::Dgram*)buf;
      return pypdsdata::Dgram::PyObject_FromPds(newDg, 0, newSize, ::_destroyDgram);

    } else {

      Py_RETURN_NONE;

    }

  } else {

    PyErr_SetString(PyExc_ValueError, "Error: XtcFilter.filter expects Xtc or Dgram object");
    return 0;

  }


}

}
