//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Pulnix_TM6740TM6740ConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "TM6740ConfigV1.h"

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
#include "../camera/FrameCoord.h"
#include "../../pdsdata_numpy.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  pypdsdata::EnumType::Enum depthEnumValues[] = {
      { "Eight_bit",  Pds::Pulnix::TM6740ConfigV1::Eight_bit },
      { "Ten_bit",    Pds::Pulnix::TM6740ConfigV1::Ten_bit },
      { 0, 0 }
  };
  pypdsdata::EnumType depthEnum ( "Depth", depthEnumValues );

  pypdsdata::EnumType::Enum binningEnumValues[] = {
      { "x1", Pds::Pulnix::TM6740ConfigV1::x1 },
      { "x2", Pds::Pulnix::TM6740ConfigV1::x2 },
      { "x4", Pds::Pulnix::TM6740ConfigV1::x4 },
      { 0, 0 }
  };
  pypdsdata::EnumType binningEnum ( "Binning", binningEnumValues );

  pypdsdata::EnumType::Enum lookupEnumValues[] = {
      { "Gamma",   Pds::Pulnix::TM6740ConfigV1::Gamma },
      { "Linear",  Pds::Pulnix::TM6740ConfigV1::Linear },
      { 0, 0 }
  };
  pypdsdata::EnumType lookupEnum ( "LookupTable", lookupEnumValues );


  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "Row_Pixels", Pds::Pulnix::TM6740ConfigV1::Row_Pixels },

        { "Column_Pixels", Pds::Pulnix::TM6740ConfigV1::Column_Pixels },

        { 0, 0 }
  };

  // methods
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, vref)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, gain_a)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, gain_b)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, gain_balance)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, shutter_width)
  ENUM_FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, output_resolution, depthEnum)
  FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, output_resolution_bits)
  ENUM_FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, horizontal_binning, binningEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, vertical_binning, binningEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Pulnix::TM6740ConfigV1, lookuptable_mode, lookupEnum)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"vref",                   vref,                    METH_NOARGS,  "" },
    {"gain_a",                 gain_a,                  METH_NOARGS,  "" },
    {"gain_b",                 gain_b,                  METH_NOARGS,  "" },
    {"gain_balance",           gain_balance,            METH_NOARGS,  "" },
    {"shutter_width",          shutter_width,           METH_NOARGS,  "Return shutter width in microseconds." },
    {"output_resolution",      output_resolution,       METH_NOARGS,  "Returns bit-depth of pixel counts (one of Depth.Eight_bit, Depth.Ten_bit)." },
    {"output_resolution_bits", output_resolution_bits,  METH_NOARGS,  "Returns bit-depth of pixel counts (in actual bits)." },
    {"horizontal_binning",     horizontal_binning,      METH_NOARGS,  "Returns horizontal re-binning of output (consecutive columns summed), one of Binning.x1, Binning.x2, etc." },
    {"vertical_binning",       vertical_binning,        METH_NOARGS,  "Returns vertical re-binning of output (consecutive rows summed), one of Binning.x1, Binning.x2, etc." },
    {"lookuptable_mode",       lookuptable_mode,        METH_NOARGS,  "one of LookupTable.Gamma, LookupTable.Linear" },

    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Pulnix::TM6740ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Pulnix::TM6740ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Depth", depthEnum.type() );
  PyDict_SetItemString( tp_dict, "Binning", binningEnum.type() );
  PyDict_SetItemString( tp_dict, "LookupTable", lookupEnum.type() );
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "TM6740ConfigV1", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Pulnix::TM6740ConfigV1* obj = pypdsdata::Pulnix::TM6740ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "pulnix.TM6740ConfigV2(vref=" << obj->vref()
      << ", gain_a=" << obj->gain_a()
      << ", gain_b=" << obj->gain_b()
      << ", bits=" << obj->output_resolution_bits()
      << ", shutter_width=" << obj->shutter_width()
      << ", ...)" ;

  return PyString_FromString( str.str().c_str() );
}

}
