//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Level...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Level.h"

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
  int Level_init( PyObject* self, PyObject* args, PyObject* kwds );
  void Level_dealloc( PyObject* self );
  PyObject* Level_str( PyObject* self );
  PyObject* Level_repr( PyObject* self );

  // type-specific methods

  PyMethodDef Level_Methods[] = {
    {0, 0, 0, 0}
   };

  char Level_doc[] = "Python class wrapping C++ Pds::Level class.\n\n"
      "This class inherits from a int type, the instances of this class\n"
      "are regular numbers with some additional niceties: repr() and str()\n"
      "functions witll print string representaion of the enum values.\n"
      "Class defines several attributes which correspond to the C++ enum values.\n\n"
      "Class constructor takes one integer numbers and creates object with\n"
      "the corresponding value of the enum.";

  PyTypeObject Level_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.xtc.Level",         /*tp_name*/
    sizeof(pypdsdata::Level),    /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    Level_dealloc,           /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    0,                       /*tp_compare*/
    Level_repr,              /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_sequence*/
    0,                       /*tp_as_mapping*/
    0,                       /*tp_hash*/
    0,                       /*tp_call*/
    Level_str,               /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    Level_doc,               /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    Level_Methods,           /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    &PyInt_Type,             /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    Level_init,              /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    Level_dealloc            /*tp_del*/
  };

}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

namespace pypdsdata {

PyTypeObject*
Level::typeObject()
{
  static bool once = true;
  if (once) {
    once = false;

    // define class attributes for enums
    PyObject* tp_dict = PyDict_New();
    for (int i = 0 ; i < Pds::Level::NumberOfLevels ; ++ i ) {
      Pds::Level::Type v = Pds::Level::Type(i);
      char* name = (char*)Pds::Level::name(v);
      PyObject* val = PyInt_FromLong(i);
      PyDict_SetItemString( tp_dict, name, val );
    }
    PyDict_SetItemString( tp_dict, "NumberOfLevels", PyInt_FromLong(Pds::Level::NumberOfLevels) );


    Level_Type.tp_dict = tp_dict;
  }

  return &::Level_Type;
}

PyObject*
Level::Level_FromInt(int value)
{
  pypdsdata::Level* ob = PyObject_New(pypdsdata::Level,&::Level_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create Level object." );
    return 0;
  }

  if ( value < 0 or value >= Pds::Level::NumberOfLevels ) {
    PyErr_SetString(PyExc_TypeError, "Error: Level out of range");
    return 0;
  }
  ob->ob_ival = value;

  return (PyObject*)ob;
}

} // namespace pypdsdata


namespace {

int
Level_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::Level* py_this = (pypdsdata::Level*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned val ;
  if ( not PyArg_ParseTuple( args, "I:Level", &val ) ) return -1;

  if ( val >= Pds::Level::NumberOfLevels ) {
    PyErr_SetString(PyExc_TypeError, "Error: Level out of range");
    return -1;
  }

  py_this->ob_ival = val;

  return 0;
}


void
Level_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

PyObject*
Level_str( PyObject* self )
{
  pypdsdata::Level* py_this = (pypdsdata::Level*) self;
  return PyString_FromString(Pds::Level::name(Pds::Level::Type(py_this->ob_ival)));
}
PyObject*
Level_repr( PyObject* self )
{
  pypdsdata::Level* py_this = (pypdsdata::Level*) self;
  return PyString_FromFormat("<Level(%ld):%s>", py_this->ob_ival,
      Pds::Level::name(Pds::Level::Type(py_this->ob_ival)));
}

}
