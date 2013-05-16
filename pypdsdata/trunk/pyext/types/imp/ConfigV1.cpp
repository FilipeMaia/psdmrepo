//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

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

  pypdsdata::EnumType::Enum registersEnumValues[] = {
    { "Range",                  Pds::Imp::ConfigV1::Range },
    { "Cal_range",              Pds::Imp::ConfigV1::Cal_range },
    { "Reset",                  Pds::Imp::ConfigV1::Reset },
    { "Bias_data",              Pds::Imp::ConfigV1::Bias_data },
    { "Cal_data",               Pds::Imp::ConfigV1::Cal_data },
    { "BiasDac_data",           Pds::Imp::ConfigV1::BiasDac_data },
    { "Cal_strobe",             Pds::Imp::ConfigV1::Cal_strobe },
    { "NumberOfSamples",        Pds::Imp::ConfigV1::NumberOfSamples },
    { "TrigDelay",              Pds::Imp::ConfigV1::TrigDelay },
    { "Adc_delay",              Pds::Imp::ConfigV1::Adc_delay },
    { "NumberOfRegisters",      Pds::Imp::ConfigV1::NumberOfRegisters },
    { 0, 0 }
  };
  pypdsdata::EnumType registersEnum ( "Registers", registersEnumValues );

  // list of enums
  pypdsdata::TypeLib::EnumEntry enums[] = {
        { "NumberOfValues",      Pds::Imp::ConfigV1::NumberOfValues },
        { "MaxNumberOfSamples",  Pds::Imp::ConfigV1::MaxNumberOfSamples },
        { 0, 0 }
  };

  // type-specific methods
  PyObject* get( PyObject* self, PyObject* args );
  PyObject* rangeHigh( PyObject* self, PyObject* args );
  PyObject* rangeLow( PyObject* self, PyObject* args );
  PyObject* defaultValue( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "get",            get,            METH_VARARGS,
        "self.get(int) -> int\n\nReturns value of the specified parameter, argument is "
        "an integer value of Registers enum." },
    { "rangeHigh",      rangeHigh,      METH_VARARGS|METH_STATIC,
        "class.rangeHigh(int) -> int\n\nReturns high range value of the specified parameter, argument is "
        "an integer value of Registers enum. This is a static method." },
    { "rangeLow",       rangeLow,       METH_VARARGS|METH_STATIC,
        "class.rangeLow(int) -> int\n\nReturns low range value of the specified parameter, argument is "
        "an integer value of Registers enum. This is a static method." },
    { "defaultValue",   defaultValue,   METH_VARARGS|METH_STATIC,
        "class.defaultValue(int) -> int\n\nReturns default value of the specified parameter, argument is "
        "an integer value of Registers enum. This is a static method." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Imp::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Imp::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "Registers", registersEnum.type() );
  pypdsdata::TypeLib::DefineEnums( type->tp_dict, ::enums );

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Imp::ConfigV1::print(std::ostream& str) const
{
  str << "imp.ConfigV1(Range=" << m_obj->get(Pds::Imp::ConfigV1::Range)
      << ", Cal_range=" << m_obj->get(Pds::Imp::ConfigV1::Cal_range)
      << ", Reset=" << m_obj->get(Pds::Imp::ConfigV1::Reset)
      << ", Bias_data=" << m_obj->get(Pds::Imp::ConfigV1::Bias_data)
      << ", Cal_data=" << m_obj->get(Pds::Imp::ConfigV1::Cal_data)
      << ", BiasDac_data=" << m_obj->get(Pds::Imp::ConfigV1::BiasDac_data)
      << ", Cal_strobe=" << m_obj->get(Pds::Imp::ConfigV1::Cal_strobe)
      << ", NumberOfSamples=" << m_obj->get(Pds::Imp::ConfigV1::NumberOfSamples)
      << ", TrigDelay=" << m_obj->get(Pds::Imp::ConfigV1::TrigDelay)
      << ", Adc_delay=" << m_obj->get(Pds::Imp::ConfigV1::Adc_delay)
      << ")";
}


namespace {

PyObject*
get( PyObject* self, PyObject* args )
{
  Pds::Imp::ConfigV1* obj = pypdsdata::Imp::ConfigV1::pdsObject( self );
  if ( not obj ) return 0;

  int arg;
  if (not PyArg_ParseTuple(args, "i:imp.ConfigV1.get", &arg)) return 0;

  return PyInt_FromLong(obj->get(Pds::Imp::ConfigV1::Registers(arg)));
}

PyObject*
rangeHigh( PyObject* self, PyObject* args )
{
  int arg;
  if (not PyArg_ParseTuple(args, "i:imp.ConfigV1.rangeHigh", &arg)) return 0;

  return PyInt_FromLong(Pds::Imp::ConfigV1::rangeHigh(Pds::Imp::ConfigV1::Registers(arg)));
}

PyObject*
rangeLow( PyObject* self, PyObject* args )
{
  int arg;
  if (not PyArg_ParseTuple(args, "i:imp.ConfigV1.rangeLow", &arg)) return 0;

  return PyInt_FromLong(Pds::Imp::ConfigV1::rangeLow(Pds::Imp::ConfigV1::Registers(arg)));
}

PyObject*
defaultValue( PyObject* self, PyObject* args )
{
  int arg;
  if (not PyArg_ParseTuple(args, "i:imp.ConfigV1.defaultValue", &arg)) return 0;

  return PyInt_FromLong(Pds::Imp::ConfigV1::defaultValue(Pds::Imp::ConfigV1::Registers(arg)));
}

}
