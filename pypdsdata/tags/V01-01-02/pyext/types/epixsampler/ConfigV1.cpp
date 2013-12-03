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
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, version)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, runTrigDelay)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, daqTrigDelay)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, daqSetting)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, adcClkHalfT)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, adcPipelineDelay)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, digitalCardId0)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, digitalCardId1)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, analogCardId0)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, analogCardId1)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, numberOfChannels)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, samplesPerChannel)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, baseClockFrequency)
  FUN0_WRAPPER(pypdsdata::EpixSampler::ConfigV1, testPatternEnable)

  PyMethodDef methods[] = {
    { "version",             version,             METH_NOARGS, "self.version() -> int\n\nReturns integer number" },
    { "runTrigDelay",        runTrigDelay,        METH_NOARGS, "self.runTrigDelay() -> int\n\nReturns integer number" },
    { "daqTrigDelay",        daqTrigDelay,        METH_NOARGS, "self.daqTrigDelay() -> int\n\nReturns integer number" },
    { "daqSetting",          daqSetting,          METH_NOARGS, "self.daqSetting() -> int\n\nReturns integer number" },
    { "adcClkHalfT",         adcClkHalfT,         METH_NOARGS, "self.adcClkHalfT() -> int\n\nReturns integer number" },
    { "adcPipelineDelay",    adcPipelineDelay,    METH_NOARGS, "self.adcPipelineDelay() -> int\n\nReturns integer number" },
    { "digitalCardId0",      digitalCardId0,      METH_NOARGS, "self.digitalCardId0() -> int\n\nReturns integer number" },
    { "digitalCardId1",      digitalCardId1,      METH_NOARGS, "self.digitalCardId1() -> int\n\nReturns integer number" },
    { "analogCardId0",       analogCardId0,       METH_NOARGS, "self.analogCardId0() -> int\n\nReturns integer number" },
    { "analogCardId1",       analogCardId1,       METH_NOARGS, "self.analogCardId1() -> int\n\nReturns integer number" },
    { "numberOfChannels",    numberOfChannels,    METH_NOARGS, "self.numberOfChannels() -> int\n\nReturns integer number" },
    { "samplesPerChannel",   samplesPerChannel,   METH_NOARGS, "self.samplesPerChannel() -> int\n\nReturns integer number" },
    { "baseClockFrequency",  baseClockFrequency,  METH_NOARGS, "self.baseClockFrequency() -> int\n\nReturns integer number" },
    { "testPatternEnable",   testPatternEnable,   METH_NOARGS, "self.testPatternEnable() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::EpixSampler::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::EpixSampler::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::EpixSampler::ConfigV1::print(std::ostream& str) const
{
  str << "EpixSampler.ConfigV1(version=" << m_obj->version()
      << ", numberOfChannels=" << m_obj->numberOfChannels()
      << ", samplesPerChannel=" << m_obj->samplesPerChannel()
      << ", baseClockFrequency=" << m_obj->baseClockFrequency()
      << ", testPatternEnable=" << int(m_obj->testPatternEnable())
      << ", runTrigDelay=" << m_obj->runTrigDelay()
      << ", daqTrigDelay=" << m_obj->daqTrigDelay()
      << ", daqSetting=" << m_obj->daqSetting()
      << ", adcClkHalfT=" << m_obj->adcClkHalfT()
      << ", adcPipelineDelay=" << m_obj->adcPipelineDelay()
      << ", ...)" ;
}
