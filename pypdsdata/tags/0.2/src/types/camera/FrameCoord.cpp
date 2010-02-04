//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FrameCoord...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "FrameCoord.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {



  // standard Python stuff
  int FrameCoord_init( PyObject* self, PyObject* args, PyObject* kwds );
  void FrameCoord_dealloc( PyObject* self );
  long FrameCoord_hash( PyObject* self );
  int FrameCoord_compare( PyObject *self, PyObject *other);
  PyObject* FrameCoord_str( PyObject *self );
  PyObject* FrameCoord_repr( PyObject *self );

  // type-specific methods
  PyObject* FrameCoord_x( PyObject* self, void* );
  PyObject* FrameCoord_y( PyObject* self, void* );

  PyMethodDef FrameCoord_Methods[] = {
    {0, 0, 0, 0}
   };

  PyGetSetDef FrameCoord_GetSet[] = {
    {"x", FrameCoord_x, 0, "column index", 0},
    {"column", FrameCoord_x, 0, "column index", 0},
    {"y", FrameCoord_y, 0, "row index", 0},
    {"row", FrameCoord_y, 0, "row index", 0},
    {0, 0, 0, 0, 0}
  };

  char FrameCoord_doc[] = "Python class wrapping C++ Pds::Camera::FrameCoord class.\n\n"
      "Constructor takes two positional arguments, same values as the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

  PyTypeObject FrameCoord_Type = {
    PyObject_HEAD_INIT(0)
    0,                       /*ob_size*/
    "pdsdata.Camera.FrameCoord",      /*tp_name*/
    sizeof(pypdsdata::Camera::FrameCoord), /*tp_basicsize*/
    0,                       /*tp_itemsize*/
    /* methods */
    FrameCoord_dealloc,      /*tp_dealloc*/
    0,                       /*tp_print*/
    0,                       /*tp_getattr*/
    0,                       /*tp_setattr*/
    FrameCoord_compare,      /*tp_compare*/
    FrameCoord_repr,         /*tp_repr*/
    0,                       /*tp_as_number*/
    0,                       /*tp_as_sequence*/
    0,                       /*tp_as_mapping*/
    FrameCoord_hash,         /*tp_hash*/
    0,                       /*tp_call*/
    FrameCoord_str,          /*tp_str*/
    PyObject_GenericGetAttr, /*tp_getattro*/
    PyObject_GenericSetAttr, /*tp_setattro*/
    0,                       /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT,      /*tp_flags*/
    FrameCoord_doc,          /*tp_doc*/
    0,                       /*tp_traverse*/
    0,                       /*tp_clear*/
    0,                       /*tp_richcompare*/
    0,                       /*tp_weaklistoffset*/
    0,                       /*tp_iter*/
    0,                       /*tp_iternext*/
    FrameCoord_Methods,      /*tp_methods*/
    0,                       /*tp_members*/
    FrameCoord_GetSet,       /*tp_getset*/
    0,                       /*tp_base*/
    0,                       /*tp_dict*/
    0,                       /*tp_descr_get*/
    0,                       /*tp_descr_set*/
    0,                       /*tp_dictoffset*/
    FrameCoord_init,         /*tp_init*/
    PyType_GenericAlloc,     /*tp_alloc*/
    PyType_GenericNew,       /*tp_new*/
    _PyObject_Del,           /*tp_free*/
    0,                       /*tp_is_gc*/
    0,                       /*tp_bases*/
    0,                       /*tp_mro*/
    0,                       /*tp_cache*/
    0,                       /*tp_subclasses*/
    0,                       /*tp_weaklist*/
    FrameCoord_dealloc       /*tp_del*/
  };

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

namespace pypdsdata {


PyTypeObject*
Camera::FrameCoord::typeObject()
{
  return &::FrameCoord_Type;
}

// makes new FrameCoord object from Pds type
PyObject*
Camera::FrameCoord::FrameCoord_FromPds(const Pds::Camera::FrameCoord& coord)
{
  pypdsdata::Camera::FrameCoord* ob = PyObject_New(pypdsdata::Camera::FrameCoord,&::FrameCoord_Type);
  if ( not ob ) {
    PyErr_SetString( PyExc_RuntimeError, "Failed to create Camera.FrameCoord object." );
    return 0;
  }

  new(&ob->m_coord) Pds::Camera::FrameCoord(coord);

  return (PyObject*)ob;
}

} // namespace pypdsdata

namespace {

int
FrameCoord_init(PyObject* self, PyObject* args, PyObject* kwds)
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;
  if ( not py_this ) {
    PyErr_SetString(PyExc_RuntimeError, "Error: self is NULL");
    return -1;
  }

  // parse arguments
  unsigned x,y;
  if ( not PyArg_ParseTuple( args, "II:Camera.FrameCoord", &x, &y ) ) return -1;

  new(&py_this->m_coord) Pds::Camera::FrameCoord( x, y );

  return 0;
}


void
FrameCoord_dealloc( PyObject* self )
{
  // deallocate ourself
  self->ob_type->tp_free(self);
}

long
FrameCoord_hash( PyObject* self )
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;
  int64_t x = py_this->m_coord.column ;
  int64_t y = py_this->m_coord.row ;
  long hash = x | ( y << 32 ) ;
  return hash;
}

int
FrameCoord_compare( PyObject* self, PyObject* other )
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;
  pypdsdata::Camera::FrameCoord* py_other = (pypdsdata::Camera::FrameCoord*) other;
  if ( py_this->m_coord.column > py_other->m_coord.column ) return 1 ;
  if ( py_this->m_coord.column < py_other->m_coord.column ) return -1 ;
  if ( py_this->m_coord.row > py_other->m_coord.row ) return 1 ;
  if ( py_this->m_coord.row < py_other->m_coord.row ) return -1 ;
  return 0 ;
}

PyObject*
FrameCoord_str( PyObject *self )
{
  return FrameCoord_repr( self );
}

PyObject*
FrameCoord_repr( PyObject *self )
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;

  char buf[32];
  snprintf( buf, sizeof buf, "Camera.FrameCoord(%d, %d)",
            py_this->m_coord.column, py_this->m_coord.row );
  return PyString_FromString( buf );
}

PyObject*
FrameCoord_x( PyObject* self, void* )
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;
  return PyInt_FromLong( py_this->m_coord.column );
}

PyObject*
FrameCoord_y( PyObject* self, void* )
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;
  return PyInt_FromLong( py_this->m_coord.row );
}

}
