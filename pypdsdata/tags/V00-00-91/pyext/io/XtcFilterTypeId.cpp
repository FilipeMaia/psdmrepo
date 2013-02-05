//--------------------------------------------------------------------------
// File and Version Information:
//  $Id$
//
// Description:
//  Class XtcFilterTypeId...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcFilterTypeId.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../types/TypeLib.h"
#include "../Xtc.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int XtcFilterTypeId_init(PyObject* self, PyObject* args, PyObject* kwds);
  PyObject* XtcFilterTypeId_call(PyObject *callable_object, PyObject *args, PyObject *kw);

  char typedoc[] = "Python class wrapping C++ XtcInput::XtcFilterTypeId class.\n\n"
      "Instances of this classes are callable objects which can be passed to XtcFilter.\n"
      "Constructor takes two lists of TypeIds (integers) - *keep* list and *discard*\n"
      "list. If keep list is empty then all object will be kept except for those\n"
      "in discard list (this also covers empty discard list). If discard list\n"
      "is empty then all objects will be discarded except for those in keep list.\n"
      "If both lists are not empty then it will keep everything in keep list but\n"
      "not in discard list.";

}

//      ----------------------------------------
//      -- Public Function Member Definitions --
//      ----------------------------------------
void
pypdsdata::XtcFilterTypeId::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_call = XtcFilterTypeId_call;
  type->tp_init = XtcFilterTypeId_init;

  BaseType::initType( "XtcFilterTypeId", module );
}

namespace {

int
XtcFilterTypeId_init(PyObject* self, PyObject* args, PyObject* kw)
{
  pypdsdata::XtcFilterTypeId* py_this = (pypdsdata::XtcFilterTypeId*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  PyObject* keep = 0;
  PyObject* discard = 0;
  if (not PyArg_ParseTuple(args, "OO:XtcFilterTypeId", &keep, &discard)) return -1;

  // check types
  if (not PySequence_Check(keep) or not PySequence_Check(discard)) {
    PyErr_SetString(PyExc_ValueError, "Error: XtcFilterTypeId expects two sequence objects");
    return -1;
  }

  // convert to C++ types
  XtcInput::XtcFilterTypeId::IdList keep_list, discard_list;
  keep_list.reserve(PySequence_Size(keep));
  discard_list.reserve(PySequence_Size(discard));
  for (int i = 0; i < PySequence_Size(keep); ++ i) {
    PyObject* elem = PySequence_GetItem(keep, i);
    // must be integer
    if (not PyInt_Check(elem)) {
      PyErr_SetString(PyExc_ValueError, "Error: XtcFilterTypeId expects lists of integers");
      return -1;
    }
    keep_list.push_back(Pds::TypeId::Type(PyInt_AS_LONG(elem)));
  }
  for (int i = 0; i < PySequence_Size(discard); ++ i) {
    PyObject* elem = PySequence_GetItem(discard, i);
    // must be integer
    if (not PyInt_Check(elem)) {
      PyErr_SetString(PyExc_ValueError, "Error: XtcFilterTypeId expects lists of integers");
      return -1;
    }
    discard_list.push_back(Pds::TypeId::Type(PyInt_AS_LONG(elem)));
  }


  new(&py_this->m_obj) XtcInput::XtcFilterTypeId(keep_list, discard_list);

  return 0;
}

PyObject*
XtcFilterTypeId_call(PyObject* self, PyObject* args, PyObject* kw)
{
  XtcInput::XtcFilterTypeId& cpp_self = pypdsdata::XtcFilterTypeId::pdsObject(self);

  // get arguments
  PyObject* obj = 0;
  if (not PyArg_ParseTuple(args, "O:XtcFilterTypeId", &obj)) return 0;

  // must be Xtc
  if (not pypdsdata::Xtc::Object_TypeCheck(obj)) {
    PyErr_SetString(PyExc_ValueError, "Error: XtcFilterTypeId expects Xtc object");
    return 0;
  }

  // get XTC
  Pds::Xtc* xtc = pypdsdata::Xtc::pdsObject(obj);

  // call C++ functor
  return PyBool_FromLong(int(cpp_self(xtc)));
}

}
