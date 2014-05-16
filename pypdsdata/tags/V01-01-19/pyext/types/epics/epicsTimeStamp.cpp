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


  namespace gs {
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::Epics::epicsTimeStamp, sec)
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::Epics::epicsTimeStamp, nsec)
  }

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"secPastEpoch",   gs::sec,     0, "integer number, seconds since Jan 1, 1990 00:00 UTC", 0},
    {"nsec",           gs::nsec,    0, "integer number, nanoseconds within second", 0},
    {0, 0, 0, 0, 0}
  };


  namespace mm {
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Epics::epicsTimeStamp, sec)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Epics::epicsTimeStamp, nsec)
  }

  PyMethodDef methods[] = {
    {"sec",       mm::sec,    METH_NOARGS,  "self.sec() -> int\n\nReturns integer number, seconds since Jan 1, 1990 00:00 UTC." },
    {"nsec",      mm::nsec,   METH_NOARGS,  "self.nsec() -> int\n\nReturns integer number, nanoseconds within second." },
    {0, 0, 0, 0}
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
  type->tp_getset = ::getset;
  type->tp_methods = ::methods;
  type->tp_init = epicsTimeStamp_init;
  type->tp_hash = epicsTimeStamp_hash;
  type->tp_compare = epicsTimeStamp_compare;

  BaseType::initType( "epicsTimeStamp", module );
}

void
pypdsdata::Epics::epicsTimeStamp::print(std::ostream& str) const
{
  str << "epicsTimeStamp(" << m_obj.sec() << ", " << m_obj.nsec() << ")";
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

  py_this->m_obj = Pds::Epics::epicsTimeStamp(sec, nsec);

  return 0;
}


long
epicsTimeStamp_hash( PyObject* self )
{
  pypdsdata::Epics::epicsTimeStamp* py_this = (pypdsdata::Epics::epicsTimeStamp*) self;
  int64_t sec = py_this->m_obj.sec() ;
  int64_t nsec = py_this->m_obj.nsec() ;
  long hash = sec*1000000000 + nsec ;
  return hash;
}

int
epicsTimeStamp_compare( PyObject* self, PyObject* other )
{
  pypdsdata::Epics::epicsTimeStamp* py_this = (pypdsdata::Epics::epicsTimeStamp*) self;
  pypdsdata::Epics::epicsTimeStamp* py_other = (pypdsdata::Epics::epicsTimeStamp*) other;
  if ( py_this->m_obj.sec() > py_other->m_obj.sec() ) return 1 ;
  if ( py_this->m_obj.sec() < py_other->m_obj.sec() ) return -1 ;
  if ( py_this->m_obj.nsec() > py_other->m_obj.nsec() ) return 1 ;
  if ( py_this->m_obj.nsec() < py_other->m_obj.nsec() ) return -1 ;
  return 0 ;
}

}
