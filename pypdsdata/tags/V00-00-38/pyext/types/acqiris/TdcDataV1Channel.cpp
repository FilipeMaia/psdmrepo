//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcDataV1Channel...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TdcDataV1Channel.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "TdcDataV1.h"
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcDataV1Channel, source, pypdsdata::Acqiris::TdcDataV1::sourceEnum())
  FUN0_WRAPPER(pypdsdata::Acqiris::TdcDataV1Channel, overflow)
  FUN0_WRAPPER(pypdsdata::Acqiris::TdcDataV1Channel, ticks)
  FUN0_WRAPPER(pypdsdata::Acqiris::TdcDataV1Channel, time)

  PyMethodDef methods[] = {
    {"source",   source,   METH_NOARGS,  "Returns TdcDataV1.Source enum value" },
    {"overflow", overflow, METH_NOARGS,  "Returns boolean value" },
    {"ticks",    ticks,    METH_NOARGS,  "Returns integer number" },
    {"time",     time,     METH_NOARGS,  "Returns floating number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TdcDataV1::Channel class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TdcDataV1Channel::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums and constants
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "Source", TdcDataV1::sourceEnum().type() );

  BaseType::initType( "TdcDataV1Channel", module );
}

void 
pypdsdata::Acqiris::TdcDataV1Channel::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.TdcDataV1Channel(None)";
  } else {  
    out << "acqiris.TdcDataV1Channel(source=" << m_obj->source()
        << ", overflow=" << (m_obj->overflow() ? "True" : "False")
        << ", ticks=" << m_obj->ticks()
        << ", time=" << m_obj->time()
        << ")" ;
  }
}
