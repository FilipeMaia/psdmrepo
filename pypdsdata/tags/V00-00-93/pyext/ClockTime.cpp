//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ClockTime...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ClockTime.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  int ClockTime_init( PyObject* self, PyObject* args, PyObject* kwds );
  long ClockTime_hash( PyObject* self );
  int ClockTime_compare( PyObject *self, PyObject *other);
  PyObject* ClockTime_str( PyObject *self );
  PyObject* ClockTime_repr( PyObject *self );

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ClockTime, seconds);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::ClockTime, nanoseconds);

  PyMethodDef methods[] = {
    { "seconds",     seconds,     METH_NOARGS, "self.seconds() -> int\n\nReturns the number of seconds" },
    { "nanoseconds", nanoseconds, METH_NOARGS, "self.nanoseconds() -> int\n\nReturns the number of nanoseconds" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::ClockTime class.\n\n"
      "Constructor takes two optional positional arguments - number of seconds\n"
      "and nanoseconds - both are integer numbers. If omitted they are\n"
      "initialized with zeros. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::ClockTime::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_init = ClockTime_init;
  type->tp_hash = ClockTime_hash;
  type->tp_compare = ClockTime_compare;
  type->tp_str = ClockTime_str;
  type->tp_repr = ClockTime_repr;

  BaseType::initType( "ClockTime", module );
}

namespace {

int
ClockTime_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned sec=0, nsec=0;
  if ( not PyArg_ParseTuple( args, "|II:ClockTime", &sec, &nsec ) ) return -1;

  if ( nsec >= 1000000000 ) {
    PyErr_SetString(PyExc_TypeError, "Error: nanoseconds value out of range");
    return -1;
  }

  new(&py_this->m_obj) Pds::ClockTime(sec, nsec);

  return 0;
}

long
ClockTime_hash( PyObject* self )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  long hash = py_this->m_obj.nanoseconds() + py_this->m_obj.seconds()*1000000000L;
  return hash;
}

int
ClockTime_compare( PyObject* self, PyObject* other )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  pypdsdata::ClockTime* py_other = (pypdsdata::ClockTime*) other;
  if ( py_this->m_obj > py_other->m_obj ) return 1 ;
  if ( py_this->m_obj == py_other->m_obj ) return 0 ;
  return -1 ;
}

PyObject*
ClockTime_str( PyObject *self )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  char buf[32];
  snprintf( buf, sizeof buf, "<T:%d.%09d>", py_this->m_obj.seconds(),
      py_this->m_obj.nanoseconds() );
  return PyString_FromString( buf );
}

PyObject*
ClockTime_repr( PyObject *self )
{
  pypdsdata::ClockTime* py_this = (pypdsdata::ClockTime*) self;
  char buf[48];
  snprintf( buf, sizeof buf, "<ClockTime(%d, %d)>", py_this->m_obj.seconds(),
      py_this->m_obj.nanoseconds() );
  return PyString_FromString( buf );
}

}
