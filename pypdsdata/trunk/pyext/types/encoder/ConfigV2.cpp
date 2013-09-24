//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV2.h"

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
      { "WRAP_FULL",   Pds::Encoder::ConfigV2::WRAP_FULL },
      { "LIMIT",       Pds::Encoder::ConfigV2::LIMIT },
      { "HALT",        Pds::Encoder::ConfigV2::HALT },
      { "WRAP_PRESET", Pds::Encoder::ConfigV2::WRAP_PRESET },
      { "END",         Pds::Encoder::ConfigV2::COUNT_END },
      { 0, 0 }
  };
  pypdsdata::EnumType countModeEnum ( "count_mode", countModeEnumValues );

  pypdsdata::EnumType::Enum quadModeEnumValues[] = {
      { "CLOCK_DIR", Pds::Encoder::ConfigV2::CLOCK_DIR },
      { "X1",        Pds::Encoder::ConfigV2::X1 },
      { "X2",        Pds::Encoder::ConfigV2::X2 },
      { "X4",        Pds::Encoder::ConfigV2::X4 },
      { "END",       Pds::Encoder::ConfigV2::QUAD_END },
      { 0, 0 }
  };
  pypdsdata::EnumType quadModeEnum ( "quad_mode", quadModeEnumValues );

  // methods
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::ConfigV2, chan_mask)
  ENUM_MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::ConfigV2, count_mode, countModeEnum)
  ENUM_MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::ConfigV2, quadrature_mode, quadModeEnum)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::ConfigV2, input_num)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::ConfigV2, input_rising)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Encoder::ConfigV2, ticks_per_sec)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"_chan_mask",       chan_mask,        0, "Integer number", 0},
    {"_count_mode",      count_mode,       0, "Integer number (:py:class:`count_mode`)", 0},
    {"_quadrature_mode", quadrature_mode,  0, "Integer number (:py:class:`quad_mode`)", 0},
    {"_input_num",       input_num,        0, "Integer number", 0},
    {"_input_rising",    input_rising,     0, "Integer number", 0},
    {"_ticks_per_sec",   ticks_per_sec,    0, "Integer number", 0},
    {0, 0, 0, 0, 0}
  };

  namespace int_methods {
  // methods
  FUN0_WRAPPER(pypdsdata::Encoder::ConfigV2, chan_mask)
  ENUM_FUN0_WRAPPER(pypdsdata::Encoder::ConfigV2, count_mode, countModeEnum)
  ENUM_FUN0_WRAPPER(pypdsdata::Encoder::ConfigV2, quadrature_mode, quadModeEnum)
  FUN0_WRAPPER(pypdsdata::Encoder::ConfigV2, input_num)
  FUN0_WRAPPER(pypdsdata::Encoder::ConfigV2, input_rising)
  FUN0_WRAPPER(pypdsdata::Encoder::ConfigV2, ticks_per_sec)
  }

  PyMethodDef methods[] = {
    { "chan_mask",     int_methods::chan_mask,  METH_NOARGS,  "self.chan_mask() -> int\n\nReturns integer number" },
    { "count_mode",    int_methods::count_mode,  METH_NOARGS,  "self.count_mode() -> enum\n\nReturns :py:class:`count_mode` enum" },
    { "count_mode",    int_methods::quadrature_mode,  METH_NOARGS,  "self.count_mode() -> enum\n\nReturns :py:class:`quad_mode` enum" },
    { "input_num",     int_methods::input_num,  METH_NOARGS,  "self.input_num() -> int\n\nReturns integer number" },
    { "input_rising",  int_methods::input_rising,  METH_NOARGS,  "self.input_rising() -> int\n\nReturns integer number" },
    { "ticks_per_sec", int_methods::ticks_per_sec,  METH_NOARGS,  "self.ticks_per_sec() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Encoder::ConfigV2 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Encoder::ConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "count_mode", countModeEnum.type() );
  PyDict_SetItemString( tp_dict, "quad_mode", quadModeEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV2", module );
}

void
pypdsdata::Encoder::ConfigV2::print(std::ostream& str) const
{
  str << "encoder.ConfigV2(chan_mask=" << m_obj->chan_mask()
      << ", count_mode=" << m_obj->count_mode()
      << ", quad_mode=" << m_obj->quadrature_mode()
      << ", input_num=" << m_obj->input_num()
      << ", ...)";
}
