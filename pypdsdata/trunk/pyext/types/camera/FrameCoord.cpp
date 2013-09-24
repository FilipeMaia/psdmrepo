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
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::Camera::FrameCoord, column)
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::Camera::FrameCoord, row)
  long FrameCoord_hash( PyObject* self );
  int FrameCoord_compare( PyObject *self, PyObject *other);

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"x",       column,   0, "column index", 0},
    {"column",  column,   0, "column index, same value as 'x'", 0},
    {"y",       row,      0, "row index", 0},
    {"row",     row,      0, "row index, same value as 'y'", 0},
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
  type->tp_getset = ::getset;
  type->tp_hash = FrameCoord_hash;
  type->tp_compare = FrameCoord_compare;

  BaseType::initType( "FrameCoord", module );
}

void
pypdsdata::Camera::FrameCoord::print(std::ostream& out) const
{
  out << "camera.FrameCoord(" << m_obj.column() << ", " << m_obj.row() << ")";
}

namespace {

long
FrameCoord_hash( PyObject* self )
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;
  uint32_t x = py_this->m_obj.column() ;
  uint32_t y = py_this->m_obj.row() ;
  long hash = x | ( y << 16 ) ;
  return hash;
}

int
FrameCoord_compare( PyObject* self, PyObject* other )
{
  pypdsdata::Camera::FrameCoord* py_this = (pypdsdata::Camera::FrameCoord*) self;
  pypdsdata::Camera::FrameCoord* py_other = (pypdsdata::Camera::FrameCoord*) other;
  if ( py_this->m_obj.column() > py_other->m_obj.column() ) return 1 ;
  if ( py_this->m_obj.column() < py_other->m_obj.column() ) return -1 ;
  if ( py_this->m_obj.row() > py_other->m_obj.row() ) return 1 ;
  if ( py_this->m_obj.row() < py_other->m_obj.row() ) return -1 ;
  return 0 ;
}

}
