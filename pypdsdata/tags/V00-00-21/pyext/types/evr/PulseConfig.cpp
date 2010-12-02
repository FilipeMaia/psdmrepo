//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrData_PulseConfig...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PulseConfig.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, pulse)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, trigger)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, set)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, clear)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, polarity)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, map_set_enable)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, map_reset_enable)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, map_trigger_enable)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, prescale)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, delay)
  FUN0_WRAPPER_EMBEDDED(pypdsdata::EvrData::PulseConfig, width)

  PyMethodDef methods[] = {
    { "pulse",              pulse,              METH_NOARGS, "internal pulse generation channel" },
    { "trigger",            trigger,            METH_NOARGS, "id of generated pulse for each mode (edge/level)" },
    { "set",                set,                METH_NOARGS, "id of generated pulse for each mode (edge/level)" },
    { "clear",              clear,              METH_NOARGS, "id of generated pulse for each mode (edge/level)" },
    { "polarity",           polarity,           METH_NOARGS, "bool: positive(negative) level" },
    { "map_set_enable",     map_set_enable,     METH_NOARGS, "bool: enable pulse generation masks" },
    { "map_reset_enable",   map_reset_enable,   METH_NOARGS, "bool: enable pulse generation masks" },
    { "map_trigger_enable", map_trigger_enable, METH_NOARGS, "bool: enable pulse generation masks" },
    { "prescale",           prescale,           METH_NOARGS, "pulse event prescale" },
    { "delay",              delay,              METH_NOARGS, "delay in 119MHz clks" },
    { "width",              width,              METH_NOARGS, "width in 119MHz clks" },
    {0, 0, 0, 0}
   };

  PyGetSetDef getset[] = {
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::EvrData::PulseConfig class.";

}

//    ----------------------------------------
//    -- Public Function Member Definitions --
//    ----------------------------------------

void
pypdsdata::EvrData::PulseConfig::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_getset = ::getset;

  BaseType::initType( "PulseConfig", module );
}
