//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_SequencerEntry...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "SequencerEntry.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::SequencerEntry, eventcode)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::SequencerEntry, delay)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "eventcode",       eventcode,       METH_NOARGS, "self.eventcode() -> int\n\nReturns integer number" },
    { "delay",           delay,           METH_NOARGS, "self.delay() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::SequencerEntry class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::SequencerEntry::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "SequencerEntry", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  const Pds::EvrData::SequencerEntry& obj = pypdsdata::EvrData::SequencerEntry::pdsObject(self);
  
  std::ostringstream str;
  str << "evr.SequencerEntry(eventcode=" << obj.eventcode()
      << ", delay=" << obj.delay() << ")"; 

  return PyString_FromString( str.str().c_str() );
}

}
