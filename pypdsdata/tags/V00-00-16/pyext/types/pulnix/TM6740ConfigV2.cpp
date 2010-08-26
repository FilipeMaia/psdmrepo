//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Pulnix_TM6740TM6740ConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TM6740ConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../EnumType.h"
#include "../../Exception.h"
#include "../TypeLib.h"
#include "../camera/FrameCoord.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum depthEnumValues[] = {
      { "Eight_bit",  Pds::Pulnix::TM6740ConfigV2::Eight_bit },
      { "Ten_bit",    Pds::Pulnix::TM6740ConfigV2::Ten_bit },
      { 0, 0 }
  };
  pypdsdata::EnumType depthEnum ( "Depth", depthEnumValues );

  pypdsdata::EnumType::Enum binningEnumValues[] = {
      { "x1", Pds::Pulnix::TM6740ConfigV2::x1 },
      { "x2", Pds::Pulnix::TM6740ConfigV2::x2 },
      { "x4", Pds::Pulnix::TM6740ConfigV2::x4 },
      { 0, 0 }
  };
  pypdsdata::EnumType binningEnum ( "Binning", binningEnumValues );

  pypdsdata::EnumType::Enum lookupEnumValues[] = {
      { "Gamma",   Pds::Pulnix::TM6740ConfigV2::Gamma },
      { "Linear",  Pds::Pulnix::TM6740ConfigV2::Linear },
      { 0, 0 }
  };
  pypdsdata::EnumType lookupEnum ( "LookupTable", lookupEnumValues );


  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "Row_Pixels", Pds::Pulnix::TM6740ConfigV2::Row_Pixels },

        { "Column_Pixels", Pds::Pulnix::TM6740ConfigV2::Column_Pixels },

        { 0, 0 }
  };

  // methods
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, vref_a)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, vref_b)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, gain_a)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, gain_b)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, gain_balance)
  ENUM_FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, output_resolution, depthEnum)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, output_resolution_bits)
  ENUM_FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, horizontal_binning, binningEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, vertical_binning, binningEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV2, lookuptable_mode, lookupEnum)

  PyMethodDef methods[] = {
    {"vref_a",                 vref_a,                  METH_NOARGS,  "" },
    {"vref_b",                 vref_b,                  METH_NOARGS,  "" },
    {"gain_a",                 gain_a,                  METH_NOARGS,  "" },
    {"gain_b",                 gain_b,                  METH_NOARGS,  "" },
    {"gain_balance",           gain_balance,            METH_NOARGS,  "" },
    {"output_resolution",      output_resolution,       METH_NOARGS,  "Returns bit-depth of pixel counts (one of Depth.Eight_bit, Depth.Ten_bit)." },
    {"output_resolution_bits", output_resolution_bits,  METH_NOARGS,  "Returns bit-depth of pixel counts (in actual bits)." },
    {"horizontal_binning",     horizontal_binning,      METH_NOARGS,  "Returns horizontal re-binning of output (consecutive columns summed), one of Binning.x1, Binning.x2, etc." },
    {"vertical_binning",       vertical_binning,        METH_NOARGS,  "Returns vertical re-binning of output (consecutive rows summed), one of Binning.x1, Binning.x2, etc." },
    {"lookuptable_mode",       lookuptable_mode,        METH_NOARGS,  "one of LookupTable.Gamma, LookupTable.Linear" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Pulnix::TM6740ConfigV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Pulnix::TM6740ConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Depth", depthEnum.type() );
  PyDict_SetItemString( tp_dict, "Binning", binningEnum.type() );
  PyDict_SetItemString( tp_dict, "LookupTable", lookupEnum.type() );
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "TM6740ConfigV2", module );
}
