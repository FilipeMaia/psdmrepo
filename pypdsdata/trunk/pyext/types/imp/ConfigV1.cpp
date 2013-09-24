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
        { "MaxNumberOfSamples",  Pds::Imp::ConfigV1::MaxNumberOfSamples },
        { 0, 0 }
  };

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, range)
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, calRange)
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, reset)
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, biasData)
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, calData)
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, biasDacData)
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, calStrobe)
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, numberOfSamples)
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, trigDelay)
  FUN0_WRAPPER(pypdsdata::Imp::ConfigV1, adcDelay)
  PyObject* get( PyObject* self, PyObject* args );
  PyObject* rangeHigh( PyObject* self, PyObject* args );
  PyObject* rangeLow( PyObject* self, PyObject* args );
  PyObject* defaultValue( PyObject* self, PyObject* args );

  PyMethodDef methods[] = {
    { "range",            range,            METH_NOARGS,   "self.range() -> int\n\nReturns integer number." },
    { "calRange",         calRange,         METH_NOARGS,   "self.calRange() -> int\n\nReturns integer number." },
    { "reset",            reset,            METH_NOARGS,   "self.reset() -> int\n\nReturns integer number." },
    { "biasData",         biasData,         METH_NOARGS,   "self.biasData() -> int\n\nReturns integer number." },
    { "calData",          calData,          METH_NOARGS,   "self.calData() -> int\n\nReturns integer number." },
    { "biasDacData",      biasDacData,      METH_NOARGS,   "self.biasDacData() -> int\n\nReturns integer number." },
    { "calStrobe",        calStrobe,        METH_NOARGS,   "self.calStrobe() -> int\n\nReturns integer number." },
    { "numberOfSamples",  numberOfSamples,  METH_NOARGS,   "self.numberOfSamples() -> int\n\nReturns integer number." },
    { "trigDelay",        trigDelay,        METH_NOARGS,   "self.trigDelay() -> int\n\nReturns integer number." },
    { "adcDelay",         adcDelay,         METH_NOARGS,   "self.adcDelay() -> int\n\nReturns integer number." },
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
  str << "imp.ConfigV1(Range=" << m_obj->range()
      << ", Cal_range=" << m_obj->calRange()
      << ", Reset=" << m_obj->reset()
      << ", Bias_data=" << m_obj->biasData()
      << ", Cal_data=" << m_obj->calData()
      << ", BiasDac_data=" << m_obj->biasDacData()
      << ", Cal_strobe=" << m_obj->calStrobe()
      << ", NumberOfSamples=" << m_obj->numberOfSamples()
      << ", TrigDelay=" << m_obj->trigDelay()
      << ", Adc_delay=" << m_obj->adcDelay()
      << ")";
}
