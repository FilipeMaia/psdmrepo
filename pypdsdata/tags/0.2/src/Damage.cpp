//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Damage...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Damage.h"

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
  int Damage_init( PyObject* self, PyObject* args, PyObject* kwds );
  void Damage_dealloc( PyObject* self );
  PyObject* Damage_str( PyObject* self );
  PyObject* Damage_repr( PyObject* self );

  // type-specific methods
  PyObject* Damage_hasDamage( PyObject* self, PyObject* args );

  PyMethodDef Damage_Methods[] = {
    { "hasDamage", Damage_hasDamage, METH_VARARGS, "Returns True if the damage bit is set, accepts values like Damage.OutOfOrder" },
    {0, 0, 0, 0}
   };

  char Damage_doc[] = "Python class wrapping C++ Pds::Damage class.\n\n"
      "This class inherits from a int type, the instances of this class\n"
      "are regular numbers with some additional niceties: repr() and str()\n"
      "functions witll print string representaion of the enum values.\n"
      "Class defines several attributes which correspond to the C++ enum values.\n\n"
      "Class constructor takes zero or one integer numbers, constructor with zero\n"
      "arguments will create no-damage object.";

  PyTypeObject Damage_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.Damage",        /*tp_name*/
    sizeof(pypdsdata::Damage),   /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    Damage_dealloc,          /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    0,                       /*tp_compare*/
    Damage_repr,             /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_sequence*/
    0,                       /*tp_as_mapping*/
    0,                       /*tp_hash*/
    0,                       /*tp_call*/
    Damage_str,              /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    Damage_doc,              /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    Damage_Methods,          /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    &PyInt_Type,             /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    Damage_init,             /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    Damage_dealloc           /*tp_del*/
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pypdsdata {

PyTypeObject*
Damage::typeObject()
{
  static bool once = true;
  if (once) {
    once = false;

    // define class attributes for enums
    PyObject* tp_dict = PyDict_New();

    PyDict_SetItemString( tp_dict, "DroppedContribution", PyInt_FromLong(Pds::Damage::DroppedContribution) );
    PyDict_SetItemString( tp_dict, "OutOfOrder", PyInt_FromLong(Pds::Damage::OutOfOrder) );
    PyDict_SetItemString( tp_dict, "OutOfSynch", PyInt_FromLong(Pds::Damage::OutOfSynch) );
    PyDict_SetItemString( tp_dict, "UserDefined", PyInt_FromLong(Pds::Damage::UserDefined) );
    PyDict_SetItemString( tp_dict, "IncompleteContribution", PyInt_FromLong(Pds::Damage::IncompleteContribution) );
    PyDict_SetItemString( tp_dict, "ContainsIncomplete", PyInt_FromLong(Pds::Damage::ContainsIncomplete) );

    PyDict_SetItemString( tp_dict, "DroppedContribution_mask", PyInt_FromLong(1<<Pds::Damage::DroppedContribution) );
    PyDict_SetItemString( tp_dict, "OutOfOrder_mask", PyInt_FromLong(1<<Pds::Damage::OutOfOrder) );
    PyDict_SetItemString( tp_dict, "OutOfSynch_mask", PyInt_FromLong(1<<Pds::Damage::OutOfSynch) );
    PyDict_SetItemString( tp_dict, "UserDefined_mask", PyInt_FromLong(1<<Pds::Damage::UserDefined) );
    PyDict_SetItemString( tp_dict, "IncompleteContribution_mask", PyInt_FromLong(1<<Pds::Damage::IncompleteContribution) );
    PyDict_SetItemString( tp_dict, "ContainsIncomplete_mask", PyInt_FromLong(1<<Pds::Damage::ContainsIncomplete) );

    Damage_Type.tp_dict = tp_dict;
  }

  return &::Damage_Type;
}

PyObject*
Damage::Damage_FromInt(int value)
{
  pypdsdata::Damage* ob = PyObject_New(pypdsdata::Damage,&::Damage_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create Damage object." );
    return 0;
  }

  ob->ob_ival = value;

  return (PyObject*)ob;
}

} // namespace pypdsdata


namespace {

int
Damage_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::Damage* py_this = (pypdsdata::Damage*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned val = 0;
  if ( not PyArg_ParseTuple( args, "|I:Damage", &val ) ) return -1;

  py_this->ob_ival = val;

  return 0;
}


void
Damage_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

PyObject*
Damage_str( PyObject* self )
{
  return Damage_repr(self);
}
PyObject*
Damage_repr( PyObject* self )
{
  pypdsdata::Damage* py_this = (pypdsdata::Damage*) self;
  return PyString_FromFormat("<Damage(%ld)>", py_this->ob_ival);
}


PyObject*
Damage_hasDamage( PyObject* self, PyObject* args )
{
  pypdsdata::Damage* py_this = (pypdsdata::Damage*) self;

  unsigned bit;
  if ( not PyArg_ParseTuple( args, "I:damage.hasDamage", &bit ) ) return 0;

  return PyBool_FromLong( py_this->ob_ival & (1 << bit) );
}

}
