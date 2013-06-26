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

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "Row_Pixels", Pds::Orca::ConfigV1::Row_Pixels },
        { "Column_Pixels", Pds::Orca::ConfigV1::Column_Pixels },
        { 0, 0 }
  };

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, mode)
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, rows)
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, cooling)
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, defect_pixel_correction_enabled)
  FUN0_WRAPPER(pypdsdata::Orca::ConfigV1, size)
  
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

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "ReadoutMode", ::readoutModeEnum.type() );
  PyDict_SetItemString( type->tp_dict, "Cooling", ::coolingEnum.type() );
  pypdsdata::TypeLib::DefineEnums( type->tp_dict, ::enums );

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Orca::ConfigV1::print(std::ostream& str) const
{
  str << "Orca.ConfigV1(mode=" << m_obj->mode()
      << ", rows=" << m_obj->rows()
      << ", cooling=" << m_obj->cooling()
      << ", defect_pixel_correction_enabled=" << (m_obj->defect_pixel_correction_enabled() ? "True" : "False")
      << ")";
}
