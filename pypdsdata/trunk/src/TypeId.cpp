//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TypeId...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TypeId.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int TypeId_init( PyObject* self, PyObject* args, PyObject* kwds );
  void TypeId_dealloc( PyObject* self );
  PyObject* TypeId_str( PyObject* self );
  PyObject* TypeId_repr( PyObject* self );
  long TypeId_hash( PyObject* self );
  int TypeId_compare( PyObject *self, PyObject *other);

  // type-specific methods
  PyObject* TypeId_value( PyObject* self );
  PyObject* TypeId_id( PyObject* self );
  PyObject* TypeId_version( PyObject* self );

  PyMethodDef TypeId_Methods[] = {
    { "value", (PyCFunction) TypeId_value, METH_NOARGS, "Returns the whole type ID number including version" },
    { "id", (PyCFunction) TypeId_id, METH_NOARGS, "Returns the type ID number without version" },
    { "version", (PyCFunction) TypeId_version, METH_NOARGS, "Returns the type ID version number" },
    {0, 0, 0, 0}
   };

  char TypeId_doc[] = "Python class wrapping C++ Pds::TypeId class.\n\n"
      "This class inherits from a int type, the instances of this class\n"
      "are regular numbers with some additional niceties: repr() and str()\n"
      "functions witll print string representaion of the enum values.\n"
      "Class defines several attributes which correspond to the C++ enum values.\n\n"
      "Class constructor takes teo optional positional arguments - type id and\n"
      "version number. If missing the values are initialized with 0.";


  PyTypeObject TypeId_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.TypeId",        /*tp_name*/
    sizeof(pypdsdata::TypeId),   /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    TypeId_dealloc,          /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    TypeId_compare,          /*tp_compare*/
    TypeId_repr,             /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_sequence*/
    0,                       /*tp_as_mapping*/
    TypeId_hash,             /*tp_hash*/
    0,                       /*tp_call*/
    TypeId_str,              /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    TypeId_doc,              /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    TypeId_Methods,          /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    TypeId_init,             /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    TypeId_dealloc           /*tp_del*/
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pypdsdata {

PyTypeObject*
TypeId::typeObject()
{
  static bool once = true;
  if (once) {
    once = false;

    // define class attributes for enums
    PyObject* tp_dict = PyDict_New();

    PyDict_SetItemString( tp_dict, "Any", PyInt_FromLong(Pds::TypeId::Any) );
    PyDict_SetItemString( tp_dict, "Id_Xtc", PyInt_FromLong(Pds::TypeId::Id_Xtc) );
    PyDict_SetItemString( tp_dict, "Id_Frame", PyInt_FromLong(Pds::TypeId::Id_Frame) );
    PyDict_SetItemString( tp_dict, "Id_AcqWaveform", PyInt_FromLong(Pds::TypeId::Id_AcqWaveform) );
    PyDict_SetItemString( tp_dict, "Id_AcqConfig", PyInt_FromLong(Pds::TypeId::Id_AcqConfig) );
    PyDict_SetItemString( tp_dict, "Id_TwoDGaussian", PyInt_FromLong(Pds::TypeId::Id_TwoDGaussian) );
    PyDict_SetItemString( tp_dict, "Id_Opal1kConfig", PyInt_FromLong(Pds::TypeId::Id_Opal1kConfig) );
    PyDict_SetItemString( tp_dict, "Id_FrameFexConfig", PyInt_FromLong(Pds::TypeId::Id_FrameFexConfig) );
    PyDict_SetItemString( tp_dict, "Id_EvrConfig", PyInt_FromLong(Pds::TypeId::Id_EvrConfig) );
    PyDict_SetItemString( tp_dict, "Id_TM6740Config", PyInt_FromLong(Pds::TypeId::Id_TM6740Config) );
    PyDict_SetItemString( tp_dict, "Id_ControlConfig", PyInt_FromLong(Pds::TypeId::Id_ControlConfig) );
    PyDict_SetItemString( tp_dict, "Id_pnCCDframe", PyInt_FromLong(Pds::TypeId::Id_pnCCDframe) );
    PyDict_SetItemString( tp_dict, "Id_pnCCDconfig", PyInt_FromLong(Pds::TypeId::Id_pnCCDconfig) );
    PyDict_SetItemString( tp_dict, "Id_Epics", PyInt_FromLong(Pds::TypeId::Id_Epics) );
    PyDict_SetItemString( tp_dict, "Id_FEEGasDetEnergy", PyInt_FromLong(Pds::TypeId::Id_FEEGasDetEnergy) );
    PyDict_SetItemString( tp_dict, "Id_EBeam", PyInt_FromLong(Pds::TypeId::Id_EBeam) );
    PyDict_SetItemString( tp_dict, "Id_PhaseCavity", PyInt_FromLong(Pds::TypeId::Id_PhaseCavity) );
    PyDict_SetItemString( tp_dict, "NumberOf", PyInt_FromLong(Pds::TypeId::NumberOf) );

    TypeId_Type.tp_dict = tp_dict;
  }

  return &::TypeId_Type;
}

PyObject*
TypeId::TypeId_FromPds(Pds::TypeId type)
{
  pypdsdata::TypeId* ob = PyObject_New(pypdsdata::TypeId,&::TypeId_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create TypeId object." );
    return 0;
  }

  new(&ob->m_typeId) Pds::TypeId( type );

  return (PyObject*)ob;
}

} // namespace pypdsdata


namespace {

int
TypeId_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned val = Pds::TypeId::Any;
  unsigned version = 0;
  if ( not PyArg_ParseTuple( args, "|II:TypeId", &val, &version ) ) return -1;

  if ( val >= Pds::TypeId::NumberOf ) {
    PyErr_SetString(PyExc_TypeError, "Error: TypeId out of range");
    return -1;
  }

  new(&py_this->m_typeId) Pds::TypeId( Pds::TypeId::Type(val), version );

  return 0;
}


void
TypeId_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

long
TypeId_hash( PyObject* self )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  long hash = py_this->m_typeId.value();
  return hash;
}

int
TypeId_compare( PyObject* self, PyObject* other )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  pypdsdata::TypeId* py_other = (pypdsdata::TypeId*) other;
  if ( py_this->m_typeId.value() > py_other->m_typeId.value() ) return 1 ;
  if ( py_this->m_typeId.value() == py_other->m_typeId.value() ) return 0 ;
  return -1 ;
}

PyObject*
TypeId_str( PyObject* self )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  if ( py_this->m_typeId.version() ) {
    return PyString_FromFormat("%s_V%d", Pds::TypeId::name(py_this->m_typeId.id()),
        py_this->m_typeId.version() );
  } else {
    return PyString_FromString( Pds::TypeId::name(py_this->m_typeId.id()) );
  }
}

PyObject*
TypeId_repr( PyObject* self )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  return PyString_FromFormat("<TypeId(%s,%d)>", Pds::TypeId::name(py_this->m_typeId.id()),
      py_this->m_typeId.version() );
}

PyObject*
TypeId_value( PyObject* self )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  return PyInt_FromLong( py_this->m_typeId.value() );
}

PyObject*
TypeId_id( PyObject* self )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  return PyInt_FromLong( py_this->m_typeId.id() );
}

PyObject*
TypeId_version( PyObject* self )
{
  pypdsdata::TypeId* py_this = (pypdsdata::TypeId*) self;
  return PyInt_FromLong( py_this->m_typeId.version() );
}

}
