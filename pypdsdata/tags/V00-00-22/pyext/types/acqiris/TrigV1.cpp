//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_TrigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TrigV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum sourceEnumValues[] = {
      { "Internal", Pds::Acqiris::TrigV1::Internal },
      { "External", Pds::Acqiris::TrigV1::External },
      { 0, 0 }
  };
  pypdsdata::EnumType sourceEnum ( "Source", sourceEnumValues );

  pypdsdata::EnumType::Enum couplingEnumValues[] = {
      { "DC",       Pds::Acqiris::TrigV1::DC },
      { "AC",       Pds::Acqiris::TrigV1::AC },
      { "HFreject", Pds::Acqiris::TrigV1::HFreject },
      { "DC50ohm",  Pds::Acqiris::TrigV1::DC50ohm },
      { "AC50ohm",  Pds::Acqiris::TrigV1::AC50ohm },
      { 0, 0 }
  };
  pypdsdata::EnumType couplingEnum ( "Coupling", couplingEnumValues );

  pypdsdata::EnumType::Enum slopeEnumValues[] = {
      { "Positive",       Pds::Acqiris::TrigV1::Positive },
      { "Negative",       Pds::Acqiris::TrigV1::Negative },
      { "OutOfWindow",    Pds::Acqiris::TrigV1::OutOfWindow },
      { "IntoWindow",     Pds::Acqiris::TrigV1::IntoWindow },
      { "HFDivide",       Pds::Acqiris::TrigV1::HFDivide },
      { "SpikeStretcher", Pds::Acqiris::TrigV1::SpikeStretcher },
      { 0, 0 }
  };
  pypdsdata::EnumType slopeEnum ( "Slope", slopeEnumValues );

  // methods
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TrigV1, coupling, couplingEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TrigV1, input, sourceEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::TrigV1, slope, slopeEnum)
  FUN0_WRAPPER(pypdsdata::Acqiris::TrigV1, level)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"coupling",     coupling,    METH_NOARGS,  "Returns integer number, one of Coupling.DC, Coupling.AC, etc." },
    {"input",        input,       METH_NOARGS,  "Returns integer number, Source.Internal or Source.External" },
    {"slope",        slope,       METH_NOARGS,  "Returns integer number, one of Slope.Positive, Slope.Negative, etc." },
    {"level",        level,       METH_NOARGS,  "Returns floating number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::TrigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::TrigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Source", sourceEnum.type() );
  PyDict_SetItemString( tp_dict, "Coupling", couplingEnum.type() );
  PyDict_SetItemString( tp_dict, "Slope", slopeEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "TrigV1", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::Acqiris::TrigV1* obj = pypdsdata::Acqiris::TrigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "acqiris.TrigV1(coupling=" << obj->coupling()
      << ", input=" << obj->input()
      << ", slope=" << obj->slope()
      << ", level=" << obj->level()
      << ")" ;
  return PyString_FromString( str.str().c_str() );
}

}
