//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_VertV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "VertV1.h"

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

  pypdsdata::EnumType::Enum couplingEnumValues[] = {
      { "GND",     Pds::Acqiris::VertV1::GND },
      { "DC",      Pds::Acqiris::VertV1::DC },
      { "AC",      Pds::Acqiris::VertV1::AC },
      { "DC50ohm", Pds::Acqiris::VertV1::DC50ohm },
      { "AC50ohm", Pds::Acqiris::VertV1::AC50ohm },
      { 0, 0 }
  };
  pypdsdata::EnumType couplingEnum ( "Coupling", couplingEnumValues );

  pypdsdata::EnumType::Enum bandwidthEnumValues[] = {
      { "None_",   Pds::Acqiris::VertV1::None },
      { "MHz25",   Pds::Acqiris::VertV1::MHz25 },
      { "MHz700",  Pds::Acqiris::VertV1::MHz700 },
      { "MHz200",  Pds::Acqiris::VertV1::MHz200 },
      { "MHz20",   Pds::Acqiris::VertV1::MHz20 },
      { "MHz35",   Pds::Acqiris::VertV1::MHz35 },
      { 0, 0 }
  };
  pypdsdata::EnumType bandwidthEnum ( "Bandwidth", bandwidthEnumValues );

  // methods
  FUN0_WRAPPER(pypdsdata::Acqiris::VertV1, fullScale)
  FUN0_WRAPPER(pypdsdata::Acqiris::VertV1, offset)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::VertV1, bandwidth, bandwidthEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Acqiris::VertV1, coupling, couplingEnum)
  FUN0_WRAPPER(pypdsdata::Acqiris::VertV1, slope)

  PyMethodDef methods[] = {
    {"fullScale",    fullScale,   METH_NOARGS,  "self.fullScale() -> float\n\nReturns floating number" },
    {"offset",       offset,      METH_NOARGS,  "self.offset() -> float\n\nReturns floating number" },
    {"coupling",     coupling,    METH_NOARGS,  "self.coupling() -> Coupling enum\n\nReturns enum value one of :py:class:`VertV1.Coupling` constants" },
    {"bandwidth",    bandwidth,   METH_NOARGS,  "self.bandwidth() -> Bandwidth enum\n\nReturns enum value one of :py:class:`VertV1.Bandwidth` constants" },
    {"slope",        slope,       METH_NOARGS,  "self.slope() -> float\n\nReturns floating number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::VertV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::VertV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Coupling", couplingEnum.type() );
  PyDict_SetItemString( tp_dict, "Bandwidth", bandwidthEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "VertV1", module );
}

void 
pypdsdata::Acqiris::VertV1::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.VertV1(None)";
  } else {  
    out << "acqiris.VertV1(fullScale=" << m_obj->fullScale()
        << ", offset=" << m_obj->offset()
        << ", coupling=" << m_obj->coupling()
        << ", bandwidth=" << m_obj->bandwidth()
        << ")" ;
  }
}
