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
  long FrameCoord_hash( PyObject* self );
  int FrameCoord_compare( PyObject *self, PyObject *other);
  PyObject* FrameCoord_str( PyObject *self );
  PyObject* FrameCoord_repr( PyObject *self );

  PyMemberDef members[] = {
    {"x",      T_USHORT, offsetof(pypdsdata::Camera::FrameCoord,m_obj.column),
       0, "column index" },
    {"column", T_USHORT, offsetof(pypdsdata::Camera::FrameCoord,m_obj.column),
       0, "column index" },
    {"y",      T_USHORT, offsetof(pypdsdata::Camera::FrameCoord,m_obj.row),
      0, "row index" },
    {"row",    T_USHORT, offsetof(pypdsdata::Camera::FrameCoord,m_obj.row),
      0, "row index" },
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Camera::FrameCoord class.\n\n"
      "Constructor takes two positional arguments, same values as the\n"
      "C++ constructor. Class implements usual comparison operators\n"
      "and hash function so that objects can be used as dictionary keys.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::Camera::FrameCoord::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_members = ::members;
  type->tp_hash = FrameCoord_hash;
  type->tp_compare = FrameCoord_compare;
  type->tp_str = FrameCoord_str;
  type->tp_repr = FrameCoord_repr;

  BaseType::initType( "FrameCoord", module );
}

namespace {

long
FrameCoord_hash( PyObject* self )
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;
  int64_t x = py_this->m_obj.column ;
  int64_t y = py_this->m_obj.row ;
  long hash = x | ( y << 32 ) ;
  return hash;
}

int
FrameCoord_compare( PyObject* self, PyObject* other )
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;
  pypdsdata::Camera::FrameCoord* py_other = (pypdsdata::Camera::FrameCoord*) other;
  if ( py_this->m_obj.column > py_other->m_obj.column ) return 1 ;
  if ( py_this->m_obj.column < py_other->m_obj.column ) return -1 ;
  if ( py_this->m_obj.row > py_other->m_obj.row ) return 1 ;
  if ( py_this->m_obj.row < py_other->m_obj.row ) return -1 ;
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
  snprintf( buf, sizeof buf, "camera.FrameCoord(%d, %d)",
            py_this->m_obj.column, py_this->m_obj.row );
  return PyString_FromString( buf );
}

}
