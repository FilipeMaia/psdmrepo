//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Dgram...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Dgram.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "Exception.h"
#include "Sequence.h"
#include "Xtc.h"
#include "ClockTime.h"
#include "pdsdata/xtc/Env.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // standard Python stuff
  PyObject* Dgram_new( PyTypeObject *subtype, PyObject *args, PyObject *kwds );

  // type-specific methods
  PyObject* Dgram_env( PyObject* self, void* );
  PyObject* Dgram_seq( PyObject* self, void* );
  PyObject* Dgram_xtc( PyObject* self, void* );
  PyObject* Dgram_getnewargs( PyObject* self, PyObject* );
  PyObject* Dgram_setClock( PyObject* self, PyObject* args );

  PyGetSetDef getset[] = {
    { "env", Dgram_env, 0, "Attribute contains the env field as an integer number.", 0 },
    { "seq", Dgram_seq, 0, "Attribute contains seq field as an :py:class:`Sequence` object.", 0 },
    { "xtc", Dgram_xtc, 0, "Attribute contains top-level :py:class:`Xtc` object.", 0 },
    {0, 0, 0, 0, 0}
  };

  PyMethodDef methods[] = {
    { "__getnewargs__",    Dgram_getnewargs, METH_NOARGS, "Pickle support" },
    { "setClock",          Dgram_setClock,   METH_VARARGS, "self.setClock(clock: ClockTime)\n\nUpdates clock value for datagram, takes :py:class:`ClockTime` object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Dgram class.\n\n"
      "Instances of his class are created by other objects, there is no\n"
      "sensible constructor for now that can be used at Python level.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Dgram::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_new = ::Dgram_new;
  type->tp_methods = ::methods;

  BaseType::initType( "Dgram", module );
}

namespace {

PyObject*
Dgram_new( PyTypeObject *subtype, PyObject *args, PyObject *kwds )
{
  // parse arguments must be a buffer object
  const char* buf;
  int bufsize;
  if ( not PyArg_ParseTuple( args, "s#:pypdsdata::Dgram", &buf, &bufsize ) ) return 0;

  // allocate memory
  pypdsdata::Dgram* py_this = static_cast<pypdsdata::Dgram*>( subtype->tp_alloc(subtype, 1) );
  if ( not py_this ) return 0;

  // initialization from buffer objects
  py_this->m_obj = (Pds::Dgram*)buf;
  py_this->m_size = bufsize;
  PyObject* parent = PyTuple_GetItem(args, 0);
  Py_INCREF(parent);
  py_this->m_parent = parent;
  py_this->m_dtor = 0;

  return py_this;
}


PyObject*
Dgram_env( PyObject* self, void* )
{
  pypdsdata::Dgram* py_this = static_cast<pypdsdata::Dgram*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return PyInt_FromLong( py_this->m_obj->env.value() );
}

PyObject*
Dgram_seq( PyObject* self, void* )
{
  pypdsdata::Dgram* py_this = static_cast<pypdsdata::Dgram*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::Sequence::PyObject_FromPds( py_this->m_obj->seq );
}

PyObject*
Dgram_xtc( PyObject* self, void* )
{
  pypdsdata::Dgram* py_this = static_cast<pypdsdata::Dgram*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::Xtc::PyObject_FromPds( &py_this->m_obj->xtc, self, py_this->m_obj->xtc.extent );
}

PyObject*
Dgram_getnewargs( PyObject* self, PyObject* )
{
  pypdsdata::Dgram* py_this = static_cast<pypdsdata::Dgram*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  size_t size = py_this->m_size;
  const char* data = (const char*)py_this->m_obj;
  PyObject* pydata = PyString_FromStringAndSize(data, size);

  PyObject* args = PyTuple_New(1);
  PyTuple_SET_ITEM(args, 0, pydata);

  return args;
}

PyObject*
Dgram_setClock( PyObject* self, PyObject* args )
{
  Pds::Dgram* pdsobj = pypdsdata::Dgram::pdsObject(self);
  if( not pdsobj ) return 0;

  // parse args
  PyObject* clockObj ;
  if ( not PyArg_ParseTuple( args, "O:xtc.Sequence.setClock", &clockObj ) ) return 0;

  if ( not pypdsdata::ClockTime::Object_TypeCheck( clockObj ) ) {
    PyErr_SetString(PyExc_TypeError, "Error: parameter is not a xtc.ClockTime object");
    return 0;      
  }
  Pds::ClockTime& clock = pypdsdata::ClockTime::pdsObject( clockObj );

  // there is no way to change clock field in datagram but there is 
  // an assignment operator
  pdsobj->seq = Pds::Sequence(clock, pdsobj->seq.stamp());

  Py_RETURN_NONE;
}

}
