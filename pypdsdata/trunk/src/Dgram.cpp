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

  // standard Python stuff
  int Dgram_init( PyObject* self, PyObject* args, PyObject* kwds );

  // type-specific methods
  PyObject* Dgram_env( PyObject* self, void* );
  PyObject* Dgram_seq( PyObject* self, void* );
  PyObject* Dgram_xtc( PyObject* self, void* );

  PyGetSetDef getset[] = {
    { "env", Dgram_env, 0, "Returns the env field as a number.", 0 },
    { "seq", Dgram_seq, 0, "Returns the seq field as an object.", 0 },
    { "xtc", Dgram_xtc, 0, "Returns top-level Xtc object.", 0 },
    {0, 0, 0, 0, 0}
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
  type->tp_init = ::Dgram_init;

  BaseType::initType( "Dgram", module );
}

namespace {

int
Dgram_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::Dgram* py_this = (pypdsdata::Dgram*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // TODO: implement initialization from buffer-like objects
  py_this->m_obj = 0;
  py_this->m_parent = 0;
  py_this->m_dtor = 0;

  return 0;
}

PyObject*
Dgram_env( PyObject* self, void* )
{
  pypdsdata::Dgram* py_this = (pypdsdata::Dgram*) self;
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return PyInt_FromLong( py_this->m_obj->env.value() );
}

PyObject*
Dgram_seq( PyObject* self, void* )
{
  pypdsdata::Dgram* py_this = (pypdsdata::Dgram*) self;
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::Sequence::PyObject_FromPds( py_this->m_obj->seq );
}

PyObject*
Dgram_xtc( PyObject* self, void* )
{
  pypdsdata::Dgram* py_this = (pypdsdata::Dgram*) self;
  if( ! py_this->m_obj ){
    PyErr_SetString(pypdsdata::exceptionType(), "Error: No Valid C++ Object");
    return 0;
  }

  return pypdsdata::Xtc::PyObject_FromPds( &py_this->m_obj->xtc, self, 0 );
}

}
