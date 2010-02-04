//--------------------------------------------------------------------------
// File and Version Information:
//  $Id$
//
// Description:
//  Class ProcInfo...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ProcInfo.h"

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
  int ProcInfo_init( PyObject* self, PyObject* args, PyObject* kwds );
  void ProcInfo_dealloc( PyObject* self );
  long ProcInfo_hash( PyObject* self );
  int ProcInfo_compare( PyObject *self, PyObject *other);
  PyObject* ProcInfo_str( PyObject *self );
  PyObject* ProcInfo_repr( PyObject *self );

  // type-specific methods
  PyObject* ProcInfo_level( PyObject* self, PyObject* );
  PyObject* ProcInfo_log( PyObject* self, PyObject* );
  PyObject* ProcInfo_phy( PyObject* self, PyObject* );
  PyObject* ProcInfo_processId( PyObject* self, PyObject* );
  PyObject* ProcInfo_ipAddr( PyObject* self, PyObject* );

  PyMethodDef ProcInfo_Methods[] = {
    { "level",      ProcInfo_level,     METH_NOARGS, "Returns source level object (Level class)" },
    { "log",        ProcInfo_log,       METH_NOARGS, "Returns logical address of data source" },
    { "phy",        ProcInfo_phy,       METH_NOARGS, "Returns physical address of data source" },
    { "processId",  ProcInfo_processId, METH_NOARGS, "Returns process ID" },
    { "ipAddr",     ProcInfo_ipAddr,    METH_NOARGS, "Returns host IP address" },
    {0, 0, 0, 0}
   };

  char ProcInfo_doc[] = "Python class wrapping C++ Pds::ProcInfo class.\n\n"
      "Constructor takes three positional arguments, same values as the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

  PyTypeObject ProcInfo_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.ProcInfo",      /*tp_name*/
    sizeof(pypdsdata::ProcInfo), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    ProcInfo_dealloc,        /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    ProcInfo_compare,        /*tp_compare*/
    ProcInfo_repr,           /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_sequence*/
    0,                       /*tp_as_mapping*/
    ProcInfo_hash,           /*tp_hash*/
    0,                       /*tp_call*/
    ProcInfo_str,            /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    ProcInfo_doc,            /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    ProcInfo_Methods,        /*tp_methods*/
    0,                       /*tp_members*/
    0,                       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    ProcInfo_init,           /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    ProcInfo_dealloc         /*tp_del*/
  };

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace pypdsdata {


PyTypeObject*
ProcInfo::typeObject()
{
  return &::ProcInfo_Type;
}

// makes new ProcInfo object from Pds type
PyObject*
ProcInfo::ProcInfo_FromPds(const Pds::ProcInfo& src)
{
  pypdsdata::ProcInfo* ob = PyObject_New(pypdsdata::ProcInfo,&::ProcInfo_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create ProcInfo object." );
    return 0;
  }

  new(&ob->m_src) Pds::ProcInfo(src);

  return (PyObject*)ob;
}

} // namespace pypdsdata

namespace {

int
ProcInfo_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned level, processId, ipAddr;
  if ( not PyArg_ParseTuple( args, "III:ProcInfo", &level, &processId, &ipAddr ) ) return -1;

  if ( level >= Pds::Level::NumberOfLevels ) {
    PyErr_SetString(PyExc_TypeError, "Error: level out of range");
    return -1;
  }

  new(&py_this->m_src) Pds::ProcInfo( Pds::Level::Type(level), processId, ipAddr );

  return 0;
}


void
ProcInfo_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

long
ProcInfo_hash( PyObject* self )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  int64_t log = py_this->m_src.log() ;
  int64_t phy = py_this->m_src.phy() ;
  long hash = log | ( phy << 32 ) ;
  return hash;
}

int
ProcInfo_compare( PyObject* self, PyObject* other )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  pypdsdata::ProcInfo* py_other = (pypdsdata::ProcInfo*) other;
  if ( py_this->m_src.log() > py_other->m_src.log() ) return 1 ;
  if ( py_this->m_src.log() < py_other->m_src.log() ) return -1 ;
  if ( py_this->m_src.phy() > py_other->m_src.phy() ) return 1 ;
  if ( py_this->m_src.phy() < py_other->m_src.phy() ) return -1 ;
  return 0 ;
}

PyObject*
ProcInfo_str( PyObject *self )
{
  return ::ProcInfo_repr( self );
}

PyObject*
ProcInfo_repr( PyObject *self )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  char buf[64];
  unsigned ip = py_this->m_src.ipAddr() ;

  snprintf( buf, sizeof buf, "<ProcInfo(%s, %d, %d.%d.%d.%d)>",
      Pds::Level::name(py_this->m_src.level()),
      py_this->m_src.processId(),
      ((ip>>24)&0xff), ((ip>>16)&0xff), ((ip>>8)&0xff), (ip&0xff) );
  return PyString_FromString( buf );
}

PyObject*
ProcInfo_level( PyObject* self, PyObject* )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  return pypdsdata::Level::Level_FromInt( py_this->m_src.level() );
}

PyObject*
ProcInfo_log( PyObject* self, PyObject* )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  return PyInt_FromLong( py_this->m_src.log() );
}

PyObject*
ProcInfo_phy( PyObject* self, PyObject* )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  return PyInt_FromLong( py_this->m_src.phy() );
}

PyObject*
ProcInfo_processId( PyObject* self, PyObject* )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  return PyInt_FromLong( py_this->m_src.processId() );
}

PyObject*
ProcInfo_ipAddr( PyObject* self, PyObject* )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  return PyInt_FromLong( py_this->m_src.ipAddr() );
}

}
