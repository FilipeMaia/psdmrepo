//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimeStamp...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TimeStamp.h"

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
  int TimeStamp_init( PyObject* self, PyObject* args, PyObject* kwds );
  long TimeStamp_hash( PyObject* self );
  int TimeStamp_compare( PyObject *self, PyObject *other);
  PyObject* TimeStamp_str( PyObject *self );
  PyObject* TimeStamp_repr( PyObject *self );

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::TimeStamp, ticks);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::TimeStamp, fiducials);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::TimeStamp, control);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::TimeStamp, vector);

  PyMethodDef methods[] = {
    { "ticks",    ticks,    METH_NOARGS,
        "self.ticks() -> int\n\n119MHz counter within the fiducial for eventcode which initiated the readout" },
    { "fiducials", fiducials, METH_NOARGS, "self.fiducials() -> int\n\nReturns the fiducials value (360Hz pulse ID) module 17-bit" },
    { "control",  control,  METH_NOARGS, "self.control() -> int\n\nInternal bits for alternate interpretation of XTC header fields" },
    { "vector",   vector,   METH_NOARGS, "self.vector() -> int\n\n15-bit seed for event-level distribution (events since Configure)" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::TimeStamp class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::TimeStamp::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_init = TimeStamp_init;
  type->tp_hash = TimeStamp_hash;
  type->tp_compare = TimeStamp_compare;
  type->tp_str = TimeStamp_str;
  type->tp_repr = TimeStamp_repr;

  BaseType::initType( "TimeStamp", module );
}


namespace {

int
TimeStamp_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  int nArgs = PyTuple_GET_SIZE( args );
  if ( nArgs == 0 ) {

    new(&py_this->m_obj) Pds::TimeStamp();

  } else {

    unsigned ticks;
    unsigned fiducials;
    unsigned vector;
    unsigned control=0;
    if ( not PyArg_ParseTuple( args, "III|I:TimeStamp", &ticks, &fiducials, &vector, &control ) ) {
      return -1;
    }

    new(&py_this->m_obj) Pds::TimeStamp(ticks, fiducials, vector, control);
  }

  return 0;
}

long
TimeStamp_hash( PyObject* self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  long hash = py_this->m_obj.fiducials();
  return hash;
}

int
TimeStamp_compare( PyObject* self, PyObject* other )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  pypdsdata::TimeStamp* py_other = (pypdsdata::TimeStamp*) other;
  if ( py_this->m_obj < py_other->m_obj ) return -1 ;
  if ( py_other->m_obj < py_this->m_obj ) return 1 ;
  return 0 ;
}

PyObject*
TimeStamp_str( PyObject *self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  char buf[32];
  snprintf( buf, sizeof buf, "<%d:%d>", py_this->m_obj.ticks(),
      py_this->m_obj.fiducials() );
  return PyString_FromString( buf );
}

PyObject*
TimeStamp_repr( PyObject *self )
{
  pypdsdata::TimeStamp* py_this = (pypdsdata::TimeStamp*) self;
  char buf[64];
  snprintf( buf, sizeof buf, "<TimeStamp(%d,%d,%d,%d)>", py_this->m_obj.ticks(),
      py_this->m_obj.fiducials(), py_this->m_obj.vector(), py_this->m_obj.control() );
  return PyString_FromString( buf );
}

}
