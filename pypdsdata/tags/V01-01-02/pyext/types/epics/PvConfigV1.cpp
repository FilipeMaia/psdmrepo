//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PvConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PvConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "epicsTimeStamp.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  namespace gs {
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::Epics::PvConfigV1, pvId)
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::Epics::PvConfigV1, description)
  MEMBER_WRAPPER_EMBEDDED_FROM_METHOD(pypdsdata::Epics::PvConfigV1, interval)
  }

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"iPvId",       gs::pvId,           0, "Integer number, PV Id", 0},
    {"sPvDesc",     gs::description,    0, "String, PV description", 0},
    {"fInterval",   gs::interval,       0, "Floating number", 0},
    {0, 0, 0, 0, 0}
  };

  namespace mm {
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Epics::PvConfigV1, pvId)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Epics::PvConfigV1, description)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::Epics::PvConfigV1, interval)
  }

  PyMethodDef methods[] = {
    {"pvId",         mm::pvId,        METH_NOARGS,  "self.pvId() -> int\n\nReturns integer number, PV ID." },
    {"description",  mm::description, METH_NOARGS,  "self.description() -> string\n\nReturns string." },
    {"interval",     mm::interval,    METH_NOARGS,  "self.interval() -> float\n\nReturns floating number." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Epics::PvConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Epics::PvConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_methods = ::methods;

  BaseType::initType( "PvConfigV1", module );
}

void
pypdsdata::Epics::PvConfigV1::print(std::ostream& str) const
{
  str << "PvConfigV1(pvId=" << m_obj.pvId()
      << ", description=\"" << m_obj.description()
      << "\", interval=" << m_obj.interval()
      << ")";
}
