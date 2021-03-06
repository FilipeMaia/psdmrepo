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
      { "WRAP_FULL",   Pds::Encoder::ConfigV2::count_mode::WRAP_FULL },
      { "LIMIT",       Pds::Encoder::ConfigV2::count_mode::LIMIT },
      { "HALT",        Pds::Encoder::ConfigV2::count_mode::HALT },
      { "WRAP_PRESET", Pds::Encoder::ConfigV2::count_mode::WRAP_PRESET },
      { "END",         Pds::Encoder::ConfigV2::count_mode::END },
      { 0, 0 }
  };
  pypdsdata::EnumType countModeEnum ( "count_mode", countModeEnumValues );

  pypdsdata::EnumType::Enum quadModeEnumValues[] = {
      { "CLOCK_DIR", Pds::Encoder::ConfigV2::quad_mode::CLOCK_DIR },
      { "X1",        Pds::Encoder::ConfigV2::quad_mode::X1 },
      { "X2",        Pds::Encoder::ConfigV2::quad_mode::X2 },
      { "X4",        Pds::Encoder::ConfigV2::quad_mode::X4 },
      { "END",       Pds::Encoder::ConfigV2::quad_mode::END },
      { 0, 0 }
  };
  pypdsdata::EnumType quadModeEnum ( "quad_mode", quadModeEnumValues );

  // methods
  MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV2, _chan_mask)
  ENUM_MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV2, _count_mode, countModeEnum)
  ENUM_MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV2, _quadrature_mode, quadModeEnum)
  MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV2, _input_num)
  MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV2, _input_rising)
  MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV2, _ticks_per_sec)
  PyObject* _repr( PyObject *self );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"_chan_mask",       _chan_mask,        0, "Integer number", 0},
    {"_count_mode",      _count_mode,       0, "Integer number (:py:class:`ConfigV2.count_mode`)", 0},
    {"_quadrature_mode", _quadrature_mode,  0, "Integer number (:py:class:`ConfigV2.quad_mode`)", 0},
    {"_input_num",       _input_num,        0, "Integer number", 0},
    {"_input_rising",    _input_rising,     0, "Integer number", 0},
    {"_ticks_per_sec",   _ticks_per_sec,    0, "Integer number", 0},
    {0, 0, 0, 0, 0}
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "count_mode", countModeEnum.type() );
  PyDict_SetItemString( tp_dict, "quad_mode", quadModeEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "ConfigV2", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Encoder::ConfigV2* obj = pypdsdata::Encoder::ConfigV2::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "encoder.ConfigV2(chan_mask=" << obj->_chan_mask
      << ", count_mode=" << obj->_count_mode
      << ", quad_mode=" << obj->_quadrature_mode
      << ", input_num=" << obj->_input_num
      << ", ...)";
  
  return PyString_FromString( str.str().c_str() );
}

}
