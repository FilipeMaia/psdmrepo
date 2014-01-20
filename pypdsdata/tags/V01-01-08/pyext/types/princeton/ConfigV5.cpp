//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV5...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV5.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

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
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, width)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, height)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, orgX)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, orgY)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, binX)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, binY)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, exposureTime)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, coolingTemp)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, gainIndex)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, readoutSpeedIndex)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, maskedHeight)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, kineticHeight)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, vsSpeed)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, infoReportInterval)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, exposureEventCode)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, numDelayShots)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV5, frameSize)

  PyMethodDef methods[] = {
    { "width",             width,             METH_NOARGS, "self.width() -> int\n\nReturns frame width" },
    { "height",            height,            METH_NOARGS, "self.height() -> int\n\nReturns fram height" },
    { "orgX",              orgX,              METH_NOARGS, "self.orgX() -> int\n\nReturns origin in X" },
    { "orgY",              orgY,              METH_NOARGS, "self.orgY() -> int\n\nReturns origin in Y" },
    { "binX",              binX,              METH_NOARGS, "self.binX() -> int\n\nReturns binning in X" },
    { "binY",              binY,              METH_NOARGS, "self.binY() -> int\n\nReturns binning in Y" },
    { "exposureTime",      exposureTime,      METH_NOARGS, "self.exposureTime() -> float\n\nReturns exposure time" },
    { "coolingTemp",       coolingTemp,       METH_NOARGS, "self.coolingTemp() -> float\n\nReturns floating number" },
    { "gainIndex",         gainIndex,         METH_NOARGS, "self.gainIndex() -> int\n\nReturns integer number" },
    { "readoutSpeedIndex", readoutSpeedIndex, METH_NOARGS, "self.readoutSpeedIndex() -> int\n\nReturns integer number" },
    { "maskedHeight",      maskedHeight,      METH_NOARGS, "self.maskedHeight() -> int\n\nReturns integer number" },
    { "kineticHeight",     kineticHeight,     METH_NOARGS, "self.kineticHeight() -> int\n\nReturns integer number" },
    { "vsSpeed",           vsSpeed,           METH_NOARGS, "self.vsSpeed() -> float\n\nReturns floating number" },
    { "infoReportInterval",infoReportInterval,METH_NOARGS, "self.infoReportInterval() -> int\n\nReturns integer number" },
    { "exposureEventCode", exposureEventCode, METH_NOARGS, "self.exposureEventCode() -> int\n\nReturns integer number" },
    { "numDelayShots",     numDelayShots,     METH_NOARGS, "self.numDelayShots() -> int\n\nReturns integer number" },
    { "frameSize",         frameSize,         METH_NOARGS, "self.frameSize() -> int\n\nCalculate the frame size based on the current ROI and binning settings" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Princeton::ConfigV5 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Princeton::ConfigV5::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV5", module );
}

void
pypdsdata::Princeton::ConfigV5::print(std::ostream& out) const
{
  out << "princeton.ConfigV5(width=" << m_obj->width()
      << ", height=" << m_obj->height()
      << ", orgX=" << m_obj->orgX()
      << ", orgY=" << m_obj->orgY()
      << ", binX=" << m_obj->binX()
      << ", binY=" << m_obj->binY()
      << ", exposureEventCode=" << m_obj->exposureEventCode()
      << ", numDelayShots=" << m_obj->numDelayShots()
      << ", ...)";
}
