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

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, width)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, height)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, orgX)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, orgY)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, binX)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, binY)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, exposureTime)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, coolingTemp)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, readoutSpeed)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, gainIndex)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, intesifierGain)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, gateDelay)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, gateWidth)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, maskedHeight)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, kineticHeight)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, vsSpeed)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, infoReportInterval)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, exposureEventCode)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, numIntegrationShots)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, frameSize)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, numPixelsX)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, numPixelsY)
  FUN0_WRAPPER(pypdsdata::Pimax::ConfigV1, numPixels)

  PyMethodDef methods[] = {
    { "width",             width,             METH_NOARGS, "self.width() -> int\n\nReturns frame width" },
    { "height",            height,            METH_NOARGS, "self.height() -> int\n\nReturns fram height" },
    { "orgX",              orgX,              METH_NOARGS, "self.orgX() -> int\n\nReturns origin in X" },
    { "orgY",              orgY,              METH_NOARGS, "self.orgY() -> int\n\nReturns origin in Y" },
    { "binX",              binX,              METH_NOARGS, "self.binX() -> int\n\nReturns binning in X" },
    { "binY",              binY,              METH_NOARGS, "self.binY() -> int\n\nReturns binning in Y" },
    { "exposureTime",      exposureTime,      METH_NOARGS, "self.exposureTime() -> float\n\nReturns exposure time" },
    { "coolingTemp",       coolingTemp,       METH_NOARGS, "self.coolingTemp() -> float\n\nReturns integer number" },
    { "readoutSpeed",      readoutSpeed,      METH_NOARGS, "self.readoutSpeed() -> float\n\nReturns readout speed" },
    { "gainIndex",         gainIndex,         METH_NOARGS, "self.gainIndex() -> int\n\nReturns integer number" },
    { "intesifierGain",    intesifierGain,    METH_NOARGS, "self.intesifierGain() -> int\n\nReturns integer number" },
    { "gateDelay",         gateDelay,         METH_NOARGS, "self.gateDelay() -> float\n\nReturns floating number" },
    { "gateWidth",         gateWidth,         METH_NOARGS, "self.gateWidth() -> float\n\nReturns floating number" },
    { "maskedHeight",      maskedHeight,      METH_NOARGS, "self.maskedHeight() -> int\n\nReturns integer number" },
    { "kineticHeight",     kineticHeight,     METH_NOARGS, "self.kineticHeight() -> int\n\nReturns integer number" },
    { "vsSpeed",           vsSpeed,           METH_NOARGS, "self.vsSpeed() -> float\n\nReturns floating number" },
    { "infoReportInterval",infoReportInterval,METH_NOARGS, "self.infoReportInterval() -> int\n\nReturns integer number" },
    { "exposureEventCode", exposureEventCode, METH_NOARGS, "self.exposureEventCode() -> int\n\nReturns integer number" },
    { "numIntegrationShots", numIntegrationShots, METH_NOARGS, "self.numIntegrationShots() -> int\n\nReturns integer number" },
    { "frameSize",         frameSize,         METH_NOARGS, "self.frameSize() -> int\n\nCalculate the frame size based on the current ROI and binning settings" },
    { "numPixelsX",        numPixelsX,        METH_NOARGS, "self.numPixelsX() -> int\n\nCalculate number of pixels based on the current ROI and binning settings" },
    { "numPixelsY",        numPixelsY,        METH_NOARGS, "self.numPixelsY() -> int\n\nCalculate number of pixels based on the current ROI and binning settings" },
    { "numPixels",         numPixels,         METH_NOARGS, "self.numPixels() -> int\n\nCalculate number of pixels based on the current ROI and binning settings" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Pimax::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Pimax::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Pimax::ConfigV1::print(std::ostream& str) const
{
  str << "Pimax.ConfigV1(width=" << m_obj->width()
      << ", height=" << m_obj->height()
      << ", orgX=" << m_obj->orgX()
      << ", orgY=" << m_obj->orgY()
      << ", binX=" << m_obj->binX()
      << ", binY=" << m_obj->binY()
      << ", exposureEventCode=" << m_obj->exposureEventCode()
      << ", ...)";
}
