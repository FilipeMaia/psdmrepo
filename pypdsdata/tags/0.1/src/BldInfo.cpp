//--------------------------------------------------------------------------
// File and Version Information:
//  $Id$
//
// Description:
//  Class BldInfo...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldInfo.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Level.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int BldInfo_init( PyObject* self, PyObject* args, PyObject* kwds );
  void BldInfo_dealloc( PyObject* self );
  long BldInfo_hash( PyObject* self );
  int BldInfo_compare( PyObject *self, PyObject *other);
  PyObject* BldInfo_str( PyObject *self );
  PyObject* BldInfo_repr( PyObject *self );

  // type-specific methods
  PyObject* BldInfo_level( PyObject* self );
  PyObject* BldInfo_log( PyObject* self );
  PyObject* BldInfo_phy( PyObject* self );
  PyObject* BldInfo_processId( PyObject* self );
  PyObject* BldInfo_type( PyObject* self );

  PyMethodDef BldInfo_Methods[] = {
    { "level", (PyCFunction) BldInfo_level, METH_NOARGS, "Returns source level object (Level class)" },
    { "log", (PyCFunction) BldInfo_log, METH_NOARGS, "Returns logical address of data source" },
    { "phy", (PyCFunction) BldInfo_phy, METH_NOARGS, "Returns physical address of data source" },
    { "processId", (PyCFunction) BldInfo_processId, METH_NOARGS, "Returns process ID" },
    { "type", (PyCFunction) BldInfo_type, METH_NOARGS, "Returns BldInfo type" },
    {0, 0, 0, 0}
   };

  char BldInfo_doc[] = "Python class wrapping C++ Pds::BldInfo class.\n\n"
      "Constructor takes two positional arguments, same values as the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

  PyTypeObject BldInfo_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.BldInfo",       /*tp_name*/
    sizeof(pypdsdata::BldInfo), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    BldInfo_dealloc,         /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    BldInfo_compare,         /*tp_compare*/
    BldInfo_repr,            /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_BldInfo*/
    0,                       /*tp_as_mapping*/
    BldInfo_hash,            /*tp_hash*/
    0,                       /*tp_call*/
    BldInfo_str,             /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    BldInfo_doc,             /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    BldInfo_Methods,         /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    BldInfo_init,            /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    BldInfo_dealloc          /*tp_del*/
  };

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace pypdsdata {


PyTypeObject*
BldInfo::typeObject()
{
  static bool once = true;
  if (once) {
    once = false;

    // define class attributes for enums
    PyObject* tp_dict = PyDict_New();
    for (int i = 0 ; i < Pds::BldInfo::NumberOf ; ++ i ) {
      Pds::BldInfo::Type v = Pds::BldInfo::Type(i);
      char* name = (char*)Pds::BldInfo::name( Pds::BldInfo(0,v) );
      PyDict_SetItemString( tp_dict, name, PyInt_FromLong(i) );
    }
    PyDict_SetItemString( tp_dict, "NumberOf", PyInt_FromLong(Pds::BldInfo::NumberOf) );

    BldInfo_Type.tp_dict = tp_dict;
  }

  return &::BldInfo_Type;
}

// makes new BldInfo object from Pds type
PyObject*
BldInfo::BldInfo_FromPds(const Pds::BldInfo& src)
{
  pypdsdata::BldInfo* ob = PyObject_New(pypdsdata::BldInfo,&::BldInfo_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create BldInfo object." );
    return 0;
  }

  new(&ob->m_src) Pds::BldInfo(src);

  return (PyObject*)ob;
}

} // namespace pypdsdata

namespace {

int
BldInfo_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned processId, type;
  if ( not PyArg_ParseTuple( args, "II:BldInfo", &processId, &type ) ) return -1;

  if ( type >= Pds::BldInfo::NumberOf ) {
    PyErr_SetString(PyExc_TypeError, "Error: type out of range");
    return -1;
  }

  new(&py_this->m_src) Pds::BldInfo( processId, Pds::BldInfo::Type(type) );

  return 0;
}


void
BldInfo_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

long
BldInfo_hash( PyObject* self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  int64_t log = py_this->m_src.log() ;
  int64_t phy = py_this->m_src.phy() ;
  long hash = log | ( phy << 32 ) ;
  return hash;
}

int
BldInfo_compare( PyObject* self, PyObject* other )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  pypdsdata::BldInfo* py_other = (pypdsdata::BldInfo*) other;
  if ( py_this->m_src.log() > py_other->m_src.log() ) return 1 ;
  if ( py_this->m_src.log() < py_other->m_src.log() ) return -1 ;
  if ( py_this->m_src.phy() > py_other->m_src.phy() ) return 1 ;
  if ( py_this->m_src.phy() < py_other->m_src.phy() ) return -1 ;
  return 0 ;
}

PyObject*
BldInfo_str( PyObject *self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return PyString_FromString( Pds::BldInfo::name(py_this->m_src) );
}

PyObject*
BldInfo_repr( PyObject *self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  char buf[32];
  snprintf( buf, sizeof buf, "<BldInfo(%d, %s)>",
      py_this->m_src.processId(),
      Pds::BldInfo::name(py_this->m_src) );
  return PyString_FromString( buf );
}

PyObject*
BldInfo_level( PyObject* self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return pypdsdata::Level::Level_FromInt( py_this->m_src.level() );
}

PyObject*
BldInfo_log( PyObject* self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return PyInt_FromLong( py_this->m_src.log() );
}

PyObject*
BldInfo_phy( PyObject* self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return PyInt_FromLong( py_this->m_src.phy() );
}

PyObject*
BldInfo_processId( PyObject* self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return PyInt_FromLong( py_this->m_src.processId() );
}

PyObject*
BldInfo_type( PyObject* self )
{
  pypdsdata::BldInfo* py_this = (pypdsdata::BldInfo*) self;
  return PyInt_FromLong( py_this->m_src.type() );
}

}
