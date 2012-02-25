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

  pypdsdata::EnumType::Enum countModeEnumValues[] = {
      { "WRAP_FULL",   Pds::Encoder::ConfigV1::count_mode::WRAP_FULL },
      { "LIMIT",       Pds::Encoder::ConfigV1::count_mode::LIMIT },
      { "HALT",        Pds::Encoder::ConfigV1::count_mode::HALT },
      { "WRAP_PRESET", Pds::Encoder::ConfigV1::count_mode::WRAP_PRESET },
      { "END",         Pds::Encoder::ConfigV1::count_mode::END },
      { 0, 0 }
  };
  pypdsdata::EnumType countModeEnum ( "count_mode", countModeEnumValues );

  pypdsdata::EnumType::Enum quadModeEnumValues[] = {
      { "CLOCK_DIR", Pds::Encoder::ConfigV1::quad_mode::CLOCK_DIR },
      { "X1",        Pds::Encoder::ConfigV1::quad_mode::X1 },
      { "X2",        Pds::Encoder::ConfigV1::quad_mode::X2 },
      { "X4",        Pds::Encoder::ConfigV1::quad_mode::X4 },
      { "END",       Pds::Encoder::ConfigV1::quad_mode::END },
      { 0, 0 }
  };
  pypdsdata::EnumType quadModeEnum ( "quad_mode", quadModeEnumValues );

  // methods
  MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV1, _chan_num)
  ENUM_MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV1, _count_mode, countModeEnum)
  ENUM_MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV1, _quadrature_mode, quadModeEnum)
  MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV1, _input_num)
  MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV1, _input_rising)
  MEMBER_WRAPPER(pypdsdata::Encoder::ConfigV1, _ticks_per_sec)
  PyObject* _repr( PyObject *self );

  PyGetSetDef getset[] = {
    {"_chan_num",         _chan_num,        0, "Integer number", 0},
    {"_count_mode",      _count_mode,       0, "Integer number", 0},
    {"_quadrature_mode", _quadrature_mode,  0, "Integer number", 0},
    {"_input_num",       _input_num,        0, "Integer number", 0},
    {"_input_rising",    _input_rising,     0, "Integer number", 0},
    {"_ticks_per_sec",   _ticks_per_sec,    0, "Integer number", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Encoder::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Encoder::ConfigV1::initType( PyObject* module )
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

  BaseType::initType( "ConfigV1", module );
}

namespace {
  
PyObject*
_repr( PyObject *self )
{
  Pds::Encoder::ConfigV1* obj = pypdsdata::Encoder::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "encoder.ConfigV1(chan_num=" << obj->_chan_num
      << ", count_mode=" << obj->_count_mode
      << ", quad_mode=" << obj->_quadrature_mode
      << ", input_num=" << obj->_input_num
      << ", ...)";
  
  return PyString_FromString( str.str().c_str() );
}

}
