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

  BaseType::initType( "SequencerEntry", module );
}

void
pypdsdata::EvrData::SequencerEntry::print(std::ostream& str) const
{
  str << "evr.SequencerEntry(eventcode=" << m_obj.eventcode()
      << ", delay=" << m_obj.delay() << ")";
}
