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
#include "pdsdata/xtc/Env.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // Dgram class supports buffer interface
  int Dgram_readbufferproc(PyObject* self, int segment, void** ptrptr);
  int Dgram_segcountproc(PyObject* self, int* lenp);

  PyBufferProcs bufferprocs = {
    Dgram_readbufferproc, // bf_getreadbuffer
    0,                    // bf_getwritebuffer
    Dgram_segcountproc,   // bf_getsegcount
    0                     // bf_getcharbuffer
  } ;

  // standard Python stuff
  PyObject* Dgram_new( PyTypeObject *subtype, PyObject *args, PyObject *kwds );

  // type-specific methods
  PyObject* Dgram_env( PyObject* self, void* );
  PyObject* Dgram_seq( PyObject* self, void* );
  PyObject* Dgram_xtc( PyObject* self, void* );
  PyObject* Dgram_getnewargs( PyObject* self, PyObject* );

  PyGetSetDef getset[] = {
    { "env", Dgram_env, 0, "Returns the env field as a number.", 0 },
    { "seq", Dgram_seq, 0, "Returns the seq field as an object.", 0 },
    { "xtc", Dgram_xtc, 0, "Returns top-level Xtc object.", 0 },
    {0, 0, 0, 0, 0}
  };

  PyMethodDef methods[] = {
    { "__getnewargs__",    Dgram_getnewargs, METH_NOARGS, "Pickle support" },
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
  type->tp_as_buffer = &::bufferprocs;

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

int
Dgram_readbufferproc(PyObject* self, int segment, void** ptrptr)
{
  pypdsdata::Dgram* py_this = static_cast<pypdsdata::Dgram*>(self);
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  *ptrptr = py_this->m_obj;
  return 0;
}

int
Dgram_segcountproc(PyObject* self, int* lenp)
{
  pypdsdata::Dgram* py_this = static_cast<pypdsdata::Dgram*>(self);
  if( ! py_this->m_obj ){
    if ( lenp ) *lenp = 0;
    return 0;
  }

  if ( lenp ) {
    *lenp = py_this->m_size;
  }
  return 1;
}

}
