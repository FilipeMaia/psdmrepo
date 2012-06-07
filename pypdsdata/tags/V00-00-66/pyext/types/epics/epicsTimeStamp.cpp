//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class epicsTimeStamp...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "epicsTimeStamp.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>
#include "python/structmember.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // standard Python stuff
  int epicsTimeStamp_init( PyObject* self, PyObject* args, PyObject* kwds );
  long epicsTimeStamp_hash( PyObject* self );
  int epicsTimeStamp_compare( PyObject *self, PyObject *other);
  PyObject* _repr( PyObject *self );

  PyMemberDef members[] = {
    {"secPastEpoch", T_UINT, offsetof(pypdsdata::Epics::epicsTimeStamp,m_obj.secPastEpoch),
       0, "integer number, seconds since 00:00 Jan 1, 1990" },
    {"nsec",         T_UINT, offsetof(pypdsdata::Epics::epicsTimeStamp,m_obj.nsec),
      0, "integer number, nanoseconds within second" },
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Epics::epicsTimeStamp class.\n\n"
      "Constructor takes two positional arguments, same values as the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::Epics::epicsTimeStamp::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_members = ::members;
  type->tp_init = epicsTimeStamp_init;
  type->tp_hash = epicsTimeStamp_hash;
  type->tp_compare = epicsTimeStamp_compare;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "epicsTimeStamp", module );
}

namespace {

int
epicsTimeStamp_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::Epics::epicsTimeStamp* py_this = (pypdsdata::Epics::epicsTimeStamp*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned sec=0, nsec=0;
  if ( not PyArg_ParseTuple( args, "|II:epicsTimeStamp", &sec, &nsec ) ) return -1;

  if ( nsec >= 1000000000 ) {
    PyErr_SetString(PyExc_TypeError, "Error: nanoseconds value out of range");
    return -1;
  }

  py_this->m_obj.secPastEpoch = sec;
  py_this->m_obj.nsec = nsec;

  return 0;
}


long
epicsTimeStamp_hash( PyObject* self )
{
  pypdsdata::Epics::epicsTimeStamp* py_this = (pypdsdata::Epics::epicsTimeStamp*) self;
  int64_t sec = py_this->m_obj.secPastEpoch ;
  int64_t nsec = py_this->m_obj.nsec ;
  long hash = sec*1000000000 + nsec ;
  return hash;
}

int
epicsTimeStamp_compare( PyObject* self, PyObject* other )
{
  pypdsdata::Epics::epicsTimeStamp* py_this = (pypdsdata::Epics::epicsTimeStamp*) self;
  pypdsdata::Epics::epicsTimeStamp* py_other = (pypdsdata::Epics::epicsTimeStamp*) other;
  if ( py_this->m_obj.secPastEpoch > py_other->m_obj.secPastEpoch ) return 1 ;
  if ( py_this->m_obj.secPastEpoch < py_other->m_obj.secPastEpoch ) return -1 ;
  if ( py_this->m_obj.nsec > py_other->m_obj.nsec ) return 1 ;
  if ( py_this->m_obj.nsec < py_other->m_obj.nsec ) return -1 ;
  return 0 ;
}

PyObject*
_repr( PyObject *self )
{
  pypdsdata::Epics::epicsTimeStamp* py_this = (pypdsdata::Epics::epicsTimeStamp*) self;

  char buf[48];
  snprintf( buf, sizeof buf, "epicsTimeStamp(%d, %d)",
            py_this->m_obj.secPastEpoch, py_this->m_obj.nsec );
  return PyString_FromString( buf );
}

}
