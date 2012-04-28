//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TdcDataV1Marker...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TdcDataV1Marker.h"

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

  pypdsdata::EnumType::Enum typeEnumValues[] = {
      { "AuxIOSwitch",    Pds::Acqiris::TdcDataV1::Marker::AuxIOSwitch },
      { "EventCntSwitch", Pds::Acqiris::TdcDataV1::Marker::EventCntSwitch },
      { "MemFullSwitch",  Pds::Acqiris::TdcDataV1::Marker::MemFullSwitch },
      { "AuxIOMarker",    Pds::Acqiris::TdcDataV1::Marker::AuxIOMarker },
      { 0, 0 }
  };
  pypdsdata::EnumType typeEnum ( "Type", typeEnumValues );

  // methods
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcDataV1Marker, source, pypdsdata::Acqiris::TdcDataV1::sourceEnum())
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TdcDataV1Marker, type, typeEnum)

  PyMethodDef methods[] = {
    {"source",   source, METH_NOARGS,  "self.source() -> Source enum\n\nReturns :py:class:`TdcDataV1.Source` enum value" },
    {"type",     type,   METH_NOARGS,  "self.type() -> Type enum\n\nReturns :py:class:`Type` enum value" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TdcDataV1::Marker class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TdcDataV1Marker::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums and constants
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "Source", TdcDataV1::sourceEnum().type() );
  PyDict_SetItemString( type->tp_dict, "Type", ::typeEnum.type() );

  BaseType::initType( "TdcDataV1Marker", module );
}

void 
pypdsdata::Acqiris::TdcDataV1Marker::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.TdcDataV1Marker(None)";
  } else {  
    out << "acqiris.TdcDataV1Marker(source=" << m_obj->source()
        << ", type=" << m_obj->type()
        << ")" ;
  }
}
