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
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int ProcInfo_init( PyObject* self, PyObject* args, PyObject* kwds );
  long ProcInfo_hash( PyObject* self );
  int ProcInfo_compare( PyObject *self, PyObject *other);
  PyObject* ProcInfo_str( PyObject *self );
  PyObject* ProcInfo_repr( PyObject *self );

  // type-specific methods
  PyObject* ProcInfo_level( PyObject* self, PyObject* );
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ProcInfo, log);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ProcInfo, phy);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ProcInfo, processId);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ProcInfo, ipAddr);

  PyMethodDef methods[] = {
    { "level",      ProcInfo_level, METH_NOARGS, "self.level() -> Level\n\nReturns source level object (:py:class:`Level` class)" },
    { "log",        log,            METH_NOARGS, "self.log() -> int\n\nReturns logical address of data source" },
    { "phy",        phy,            METH_NOARGS, "self.phy() -> int\n\nReturns physical address of data source" },
    { "processId",  processId,      METH_NOARGS, "self.processId() -> int\n\nReturns process ID" },
    { "ipAddr",     ipAddr,         METH_NOARGS, "self.ipAddr() -> int\n\nReturns host IP address" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::ProcInfo class.\n\n"
      "Constructor takes three positional arguments, same values as the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::ProcInfo::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_init = ProcInfo_init;
  type->tp_hash = ProcInfo_hash;
  type->tp_compare = ProcInfo_compare;
  type->tp_str = ProcInfo_str;
  type->tp_repr = ProcInfo_repr;

  BaseType::initType( "ProcInfo", module );
}

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

  new(&py_this->m_obj) Pds::ProcInfo( Pds::Level::Type(level), processId, ipAddr );

  return 0;
}

long
ProcInfo_hash( PyObject* self )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  int64_t log = py_this->m_obj.log() ;
  int64_t phy = py_this->m_obj.phy() ;
  long hash = log | ( phy << 32 ) ;
  return hash;
}

int
ProcInfo_compare( PyObject* self, PyObject* other )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  pypdsdata::ProcInfo* py_other = (pypdsdata::ProcInfo*) other;
  if ( py_this->m_obj.log() > py_other->m_obj.log() ) return 1 ;
  if ( py_this->m_obj.log() < py_other->m_obj.log() ) return -1 ;
  if ( py_this->m_obj.phy() > py_other->m_obj.phy() ) return 1 ;
  if ( py_this->m_obj.phy() < py_other->m_obj.phy() ) return -1 ;
  return 0 ;
}

PyObject*
ProcInfo_str( PyObject *self )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  char buf[64];
  unsigned ip = py_this->m_obj.ipAddr() ;

  snprintf( buf, sizeof buf, "ProcInfo(%s, %d, %d.%d.%d.%d)",
      Pds::Level::name(py_this->m_obj.level()),
      py_this->m_obj.processId(),
      ((ip>>24)&0xff), ((ip>>16)&0xff), ((ip>>8)&0xff), (ip&0xff) );
  return PyString_FromString( buf );
}

PyObject*
ProcInfo_repr( PyObject *self )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  char buf[64];
  unsigned ip = py_this->m_obj.ipAddr() ;

  snprintf( buf, sizeof buf, "<ProcInfo(%s, %d, %d.%d.%d.%d)>",
      Pds::Level::name(py_this->m_obj.level()),
      py_this->m_obj.processId(),
      ((ip>>24)&0xff), ((ip>>16)&0xff), ((ip>>8)&0xff), (ip&0xff) );
  return PyString_FromString( buf );
}

PyObject*
ProcInfo_level( PyObject* self, PyObject* )
{
  pypdsdata::ProcInfo* py_this = (pypdsdata::ProcInfo*) self;
  return pypdsdata::Level::Level_FromInt( py_this->m_obj.level() );
}

}
