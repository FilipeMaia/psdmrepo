//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TransitionId...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TransitionId.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int TransitionId_init( PyObject* self, PyObject* args, PyObject* kwds );
  void TransitionId_dealloc( PyObject* self );
  PyObject* TransitionId_str( PyObject* self );
  PyObject* TransitionId_repr( PyObject* self );

  // type-specific methods

  PyMethodDef TransitionId_Methods[] = {
    {0, 0, 0, 0}
   };

  char TransitionId_doc[] = "Python class wrapping C++ Pds::TransitionId class.\n\n"
      "This class inherits from a int type, the instances of this class\n"
      "are regular numbers with some additional niceties: repr() and str()\n"
      "functions witll print string representaion of the enum values.\n"
      "Class defines several attributes which correspond to the C++ enum values.\n\n"
      "Class constructor takes zero or one integer numbers, constructor with zero\n"
      "arguments will create Unknown transition id, constructor with one argument\n"
      "creates object with the corresponding value of the enum.";

  PyTypeObject TransitionId_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.TransitionId",  /*tp_name*/
    sizeof(pypdsdata::TransitionId), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    TransitionId_dealloc,    /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    0,                       /*tp_compare*/
    TransitionId_repr,       /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_sequence*/
    0,                       /*tp_as_mapping*/
    0,                       /*tp_hash*/
    0,                       /*tp_call*/
    TransitionId_str,        /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    TransitionId_doc,        /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    TransitionId_Methods,    /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    &PyInt_Type,             /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    TransitionId_init,       /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    TransitionId_dealloc     /*tp_del*/
  };

}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace pypdsdata {

PyTypeObject*
TransitionId::typeObject()
{
  static bool once = true;
  if (once) {
    once = false;

    // define class attributes for enums
    PyObject* tp_dict = PyDict_New();
    for (int i = 0 ; i < Pds::TransitionId::NumberOf ; ++ i ) {
      Pds::TransitionId::Value v = Pds::TransitionId::Value(i);
      char* name = (char*)Pds::TransitionId::name(v);
      PyObject* val = PyInt_FromLong(i);
      PyDict_SetItemString( tp_dict, name, val );
    }
    PyDict_SetItemString( tp_dict, "NumberOf", PyInt_FromLong(Pds::TransitionId::NumberOf) );

    TransitionId_Type.tp_dict = tp_dict;
  }

  return &::TransitionId_Type;
}

PyObject*
TransitionId::TransitionId_FromInt(int value)
{
  pypdsdata::TransitionId* ob = PyObject_New(pypdsdata::TransitionId,&::TransitionId_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create TransitionId object." );
    return 0;
  }

  if ( value < 0 or value >= Pds::TransitionId::NumberOf ) {
    PyErr_SetString(PyExc_TypeError, "Error: TransitionId out of range");
    return 0;
  }
  ob->ob_ival = value;

  return (PyObject*)ob;
}

} // namespace pypdsdata


namespace {

int
TransitionId_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::TransitionId* py_this = (pypdsdata::TransitionId*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned val = Pds::TransitionId::Unknown;
  if ( not PyArg_ParseTuple( args, "|I:TransitionId", &val ) ) return -1;

  if ( val >= Pds::TransitionId::NumberOf ) {
    PyErr_SetString(PyExc_TypeError, "Error: TransitionId out of range");
    return -1;
  }

  py_this->ob_ival = val;

  return 0;
}


void
TransitionId_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

PyObject*
TransitionId_str( PyObject* self )
{
  pypdsdata::TransitionId* py_this = (pypdsdata::TransitionId*) self;
  return PyString_FromString(Pds::TransitionId::name(Pds::TransitionId::Value(py_this->ob_ival)));
}
PyObject*
TransitionId_repr( PyObject* self )
{
  pypdsdata::TransitionId* py_this = (pypdsdata::TransitionId*) self;
  return PyString_FromFormat("<TransitionId(%ld):%s>", py_this->ob_ival,
      Pds::TransitionId::name(Pds::TransitionId::Value(py_this->ob_ival)));
}

}
