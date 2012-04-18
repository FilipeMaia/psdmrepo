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

  pypdsdata::EnumType::Enum inputModeEnumValues[] = {
      { "InputMode_Differential", Pds::Gsc16ai::ConfigV1::InputMode_Differential },
      { "InputMode_Zero",         Pds::Gsc16ai::ConfigV1::InputMode_Zero },
      { "InputMode_Vref",         Pds::Gsc16ai::ConfigV1::InputMode_Vref },
      { 0, 0 }
  };
  pypdsdata::EnumType inputModeEnum ( "InputMode", inputModeEnumValues );
  
  pypdsdata::EnumType::Enum voltageRangeEnumValues[] = {
      { "VoltageRange_10V",  Pds::Gsc16ai::ConfigV1::VoltageRange_10V },
      { "VoltageRange_5V",   Pds::Gsc16ai::ConfigV1::VoltageRange_5V },
      { "VoltageRange_2_5V", Pds::Gsc16ai::ConfigV1::VoltageRange_2_5V },
      { 0, 0 }
  };
  pypdsdata::EnumType voltageRangeEnum ( "VoltageRange", voltageRangeEnumValues );
  
  pypdsdata::EnumType::Enum triggerModeEnumValues[] = {
      { "TriggerMode_ExtPos", Pds::Gsc16ai::ConfigV1::TriggerMode_ExtPos },
      { "TriggerMode_ExtNeg", Pds::Gsc16ai::ConfigV1::TriggerMode_ExtNeg },
      { "TriggerMode_IntClk", Pds::Gsc16ai::ConfigV1::TriggerMode_IntClk },
      { 0, 0 }
  };
  pypdsdata::EnumType triggerModeEnum ( "TriggerMode", triggerModeEnumValues );
  
  pypdsdata::EnumType::Enum dataFormatEnumValues[] = {
      { "DataFormat_TwosComplement", Pds::Gsc16ai::ConfigV1::DataFormat_TwosComplement },
      { "DataFormat_OffsetBinary",  Pds::Gsc16ai::ConfigV1::DataFormat_OffsetBinary },
      { 0, 0 }
  };
  pypdsdata::EnumType dataFormatEnum ( "DataFormat", dataFormatEnumValues );
  
  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
      
        { "LowestChannel", Pds::Gsc16ai::ConfigV1::LowestChannel },
        { "HighestChannel", Pds::Gsc16ai::ConfigV1::HighestChannel },

        { "LowestFps", Pds::Gsc16ai::ConfigV1::LowestFps },
        { "HighestFps", Pds::Gsc16ai::ConfigV1::HighestFps },

        { 0, 0 }
  };

  // type-specific methods
  ENUM_FUN0_WRAPPER(pypdsdata::Gsc16ai::ConfigV1, voltageRange, voltageRangeEnum)
  FUN0_WRAPPER(pypdsdata::Gsc16ai::ConfigV1, firstChan)
  FUN0_WRAPPER(pypdsdata::Gsc16ai::ConfigV1, lastChan)
  ENUM_FUN0_WRAPPER(pypdsdata::Gsc16ai::ConfigV1, inputMode, inputModeEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Gsc16ai::ConfigV1, triggerMode, triggerModeEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Gsc16ai::ConfigV1, dataFormat, dataFormatEnum)
  FUN0_WRAPPER(pypdsdata::Gsc16ai::ConfigV1, fps)
  FUN0_WRAPPER(pypdsdata::Gsc16ai::ConfigV1, autocalibEnable)
  FUN0_WRAPPER(pypdsdata::Gsc16ai::ConfigV1, timeTagEnable)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "voltageRange",      voltageRange,     METH_NOARGS, "self.voltageRange() -> VoltageRange enum\n\nReturns :py:class:`VoltageRange` enum" },
    { "firstChan",         firstChan,        METH_NOARGS, "self.firstChan() -> int\n\nReturns integer number" },
    { "lastChan",          lastChan,         METH_NOARGS, "self.lastChan() -> int\n\nReturns integer number" },
    { "inputMode",         inputMode,        METH_NOARGS, "self.inputMode() -> InputMode enum\n\nReturns :py:class:`InputMode` enum" },
    { "triggerMode",       triggerMode,      METH_NOARGS, "self.triggerMode() -> TriggerMode enum\n\nReturns :py:class:`TriggerMode` enum" },
    { "dataFormat",        dataFormat,       METH_NOARGS, "self.dataFormat() -> DataFormat enum\n\nReturns :py:class:`DataFormat` enum" },
    { "fps",               fps,              METH_NOARGS, "self.fps() -> int\n\nReturns integer number" },
    { "autocalibEnable",   autocalibEnable,  METH_NOARGS, "self.autocalibEnable() -> bool\n\nReturns boolean" },
    { "timeTagEnable",     timeTagEnable,    METH_NOARGS, "self.timeTagEnable() -> bool\n\nReturns boolean" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Gsc16ai::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Gsc16ai::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "InputMode", inputModeEnum.type() );
  PyDict_SetItemString( tp_dict, "VoltageRange", voltageRangeEnum.type() );
  PyDict_SetItemString( tp_dict, "TriggerMode", triggerModeEnum.type() );
  PyDict_SetItemString( tp_dict, "DataFormat", dataFormatEnum.type() );
  pypdsdata::TypeLib::DefineEnums( tp_dict, ::enums );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV1", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Gsc16ai::ConfigV1* obj = pypdsdata::Gsc16ai::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "Gsc16ai.ConfigV1(voltageRange=" << obj->voltageRange()
      << ", firstChan=" << obj->firstChan()
      << ", lastChan=" << obj->lastChan()
      << ", inputMode=" << obj->inputMode()
      << ", triggerMode=" << obj->triggerMode()
      << ", dataFormat=" << obj->dataFormat()
      << ", fps=" << obj->fps()
      << ", autocalibEnable=" << (obj->autocalibEnable() ? 1 : 0)
      << ", timeTagEnable=" << (obj->timeTagEnable() ? 1 : 0)
      << ")" ;

  return PyString_FromString( str.str().c_str() );
}

}
