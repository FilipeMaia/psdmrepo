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
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"fullScale",    fullScale,   METH_NOARGS,  "Returns floating number" },
    {"offset",       offset,      METH_NOARGS,  "Returns floating number" },
    {"coupling",     coupling,    METH_NOARGS,  "Returns integer number, one of Coupling.GND, Coupling.DC, Coupling.AC, etc." },
    {"bandwidth",    bandwidth,   METH_NOARGS,  "Returns integer number, one of Bandwidth.None_, Bandwidth.MHz25, Bandwidth.MHz700, etc." },
    {"slope",        slope,       METH_NOARGS,  "Returns floating number" },
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Coupling", couplingEnum.type() );
  PyDict_SetItemString( tp_dict, "Bandwidth", bandwidthEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "VertV1", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::Acqiris::VertV1* obj = pypdsdata::Acqiris::VertV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "acqiris.VertV1(fullScale=" << obj->fullScale()
      << ", offset=" << obj->offset()
      << ", coupling=" << obj->coupling()
      << ", bandwidth=" << obj->bandwidth()
      << ")" ;
  return PyString_FromString( str.str().c_str() );
}

}
