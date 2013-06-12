//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_DataV3_FIFOEvent...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataV3_FIFOEvent.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // methods
  MEMBER_WRAPPER_EMBEDDED(pypdsdata::EvrData::DataV3_FIFOEvent, TimestampHigh)
  MEMBER_WRAPPER_EMBEDDED(pypdsdata::EvrData::DataV3_FIFOEvent, TimestampLow)
  MEMBER_WRAPPER_EMBEDDED(pypdsdata::EvrData::DataV3_FIFOEvent, EventCode)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"TimestampHigh",  TimestampHigh,     0, "Integer number", 0},
    {"TimestampLow",   TimestampLow,      0, "Integer number", 0},
    {"EventCode",      EventCode,         0, "Integer number", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::DataV3::FIFOEvent class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::DataV3_FIFOEvent::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "DataV3_FIFOEvent", module );
}
