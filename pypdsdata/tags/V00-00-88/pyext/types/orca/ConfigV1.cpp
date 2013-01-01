//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

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

  pypdsdata::EnumType::Enum readoutModeEnumValues[] = {
      { "x1",       Pds::Orca::ConfigV1::x1 },
      { "x2",       Pds::Orca::ConfigV1::x2 },
      { "x4",       Pds::Orca::ConfigV1::x4 },
      { "Subarray", Pds::Orca::ConfigV1::Subarray },
      { 0, 0 }
  };
  pypdsdata::EnumType readoutModeEnum ( "ReadoutMode", readoutModeEnumValues );

  pypdsdata::EnumType::Enum coolingEnumValues[] = {
      { "Off",      Pds::Orca::ConfigV1::Off },
      { "On",       Pds::Orca::ConfigV1::On },
      { "Max",      Pds::Orca::ConfigV1::Max },
      { 0, 0 }
  };
  pypdsdata::EnumType coolingEnum ( "Cooling", coolingEnumValues );

// type-specific methods
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, mode)
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, rows)
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, cooling)
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, defect_pixel_correction_enabled)
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, size)
  PyObject* _repr( PyObject *self );
  
  PyMethodDef methods[] = {
    { "mode",             mode,             METH_NOARGS, "self.mode() -> ReadoutMode\n\nReturns readout mode" },
    { "rows",             rows,             METH_NOARGS, "self.rows() -> int\n\nReturns integer number" },
    { "cooling",          cooling,          METH_NOARGS, "self.cooling() -> Cooling\n\nReturns cooling mode" },
    { "defect_pixel_correction_enabled",  defect_pixel_correction_enabled, METH_NOARGS, 
                            "self.defect_pixel_correction_enabled() -> bool\n\nReturns True if correction is enabled" },
    { "size",              size,            METH_NOARGS, "self.size() -> int\n\nReturns size of this object" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Orca::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Orca::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "ReadoutMode", readoutModeEnum.type() );
  PyDict_SetItemString( tp_dict, "Cooling", coolingEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV1", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::Orca::ConfigV1* obj = pypdsdata::Orca::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "Orca.ConfigV1(mode=" << obj->mode()
      << ", rows=" << obj->rows()
      << ", cooling=" << obj->cooling()
      << ", defect_pixel_correction_enabled=" << (obj->defect_pixel_correction_enabled() ? "True" : "False")
      << ")";

  return PyString_FromString( str.str().c_str() );
}

}
