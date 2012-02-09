//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcDataV1Common...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TdcDataV1Common.h"

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
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcDataV1Common, source, pypdsdata::Acqiris::TdcDataV1::sourceEnum())
  FUN0_WRAPPER(pypdsdata::Acqiris::TdcDataV1Common, overflow)
  FUN0_WRAPPER(pypdsdata::Acqiris::TdcDataV1Common, nhits)

  PyMethodDef methods[] = {
    {"source",   source,   METH_NOARGS,  "self.source() -> Source enum\n\nReturns TdcDataV1.Source enum value" },
    {"overflow", overflow, METH_NOARGS,  "self.overflow() -> bool\n\nReturns boolean value" },
    {"nhits",    nhits,    METH_NOARGS,  "self.nhits() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TdcDataV1::Common class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TdcDataV1Common::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums and constants
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "Source", TdcDataV1::sourceEnum().type() );

  BaseType::initType( "TdcDataV1Common", module );
}

void 
pypdsdata::Acqiris::TdcDataV1Common::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.TdcDataV1Common(None)";
  } else {  
    out << "acqiris.TdcDataV1Common(source=" << m_obj->source()
        << ", overflow=" << (m_obj->overflow() ? "True" : "False")
        << ", nhits=" << m_obj->nhits()
        << ")" ;
  }
}
