//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class UsdUsb_ConfigV1...
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

  pypdsdata::EnumType::Enum countModeEnumValues[] = {
      { "WRAP_FULL",   Pds::UsdUsb::ConfigV1::WRAP_FULL },
      { "LIMIT",       Pds::UsdUsb::ConfigV1::LIMIT },
      { "HALT",        Pds::UsdUsb::ConfigV1::HALT },
      { "WRAP_PRESET", Pds::UsdUsb::ConfigV1::WRAP_PRESET },
      { 0, 0 }
  };
  pypdsdata::EnumType countModeEnum ( "Count_Mode", countModeEnumValues );

  pypdsdata::EnumType::Enum quadModeEnumValues[] = {
      { "CLOCK_DIR",   Pds::UsdUsb::ConfigV1::CLOCK_DIR },
      { "X1",          Pds::UsdUsb::ConfigV1::X1 },
      { "X2",          Pds::UsdUsb::ConfigV1::X2 },
      { "X4",          Pds::UsdUsb::ConfigV1::X4 },
      { 0, 0 }
  };
  pypdsdata::EnumType quadModeEnum ( "Quad_Mode", quadModeEnumValues );

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "NCHANNELS", Pds::UsdUsb::ConfigV1::NCHANNELS },
        { 0, 0 }
  };

  // methods
  FUN0_WRAPPER(pypdsdata::UsdUsb::ConfigV1, counting_mode)
  FUN0_WRAPPER(pypdsdata::UsdUsb::ConfigV1, quadrature_mode)

  PyMethodDef methods[] = {
    {"counting_mode",   counting_mode,   METH_NOARGS,
        "self.counting_mode() -> list of ints\n\nReturns list of NCHANNEL numbers corresponding to Count_Mode values" },
    {"quadrature_mode", quadrature_mode, METH_NOARGS,
        "self.quadrature_mode() -> list of ints\n\nReturns list of NCHANNEL numbers corresponding to Quad_Mode values" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::UsdUsb::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::UsdUsb::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Count_Mode", countModeEnum.type() );
  PyDict_SetItemString( tp_dict, "Quad_Mode", quadModeEnum.type() );
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::UsdUsb::ConfigV1::print(std::ostream& str) const
{
  str << "usdusb.ConfigV1(counting_modes=" << m_obj->counting_mode()
      << ", quadrature_modes=" << m_obj->quadrature_mode()
      << ")";
}
