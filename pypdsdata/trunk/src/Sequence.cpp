//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Sequence...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "Sequence.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <new>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "ClockTime.h"
#include "EnumType.h"
#include "TimeStamp.h"
#include "TransitionId.h"
#include "types/TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum typeEnumValues[] = {
      { "Event",      Pds::Sequence::Event },
      { "Occurrence", Pds::Sequence::Occurrence },
      { "Marker",     Pds::Sequence::Marker },
      { 0, 0 }
  };
  pypdsdata::EnumType typeEnum ( "Type", typeEnumValues );

  // type-specific methods
  ENUM_FUN0_WRAPPER_EMBEDDED(pypdsdata::Sequence, type, typeEnum);
  PyObject* Sequence_service( PyObject* self, PyObject* );
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Sequence, isExtended);
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Sequence, isEvent);
  PyObject* Sequence_clock( PyObject* self, PyObject* );
  PyObject* Sequence_stamp( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    { "type",       type,          METH_NOARGS,
        "Returns the type of this sequence, one of Type.Event, Type.Occurrence, or Type.Marker" },
    { "service",    Sequence_service, METH_NOARGS, "Returns the TransitionId type" },
    { "isExtended", isExtended,     METH_NOARGS, "Returns True for extended sequence" },
    { "isEvent",    isEvent,        METH_NOARGS, "Returns True for event sequence" },
    { "clock",      Sequence_clock, METH_NOARGS, "Returns clock value for sequence" },
    { "stamp",      Sequence_stamp, METH_NOARGS, "Returns timestamp value for sequence" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Sequence class.";

}


//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Sequence::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Type", typeEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "Sequence", module );
}

namespace {

PyObject*
Sequence_service( PyObject* self, PyObject* )
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  return pypdsdata::TransitionId::TransitionId_FromInt( py_this->m_obj.service() );
}

PyObject*
Sequence_clock( PyObject* self, PyObject* )
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  return pypdsdata::ClockTime::PyObject_FromPds( py_this->m_obj.clock() );
}

PyObject*
Sequence_stamp( PyObject* self, PyObject* )
{
  pypdsdata::Sequence* py_this = (pypdsdata::Sequence*) self;
  return pypdsdata::TimeStamp::PyObject_FromPds( py_this->m_obj.stamp() );
}

}
