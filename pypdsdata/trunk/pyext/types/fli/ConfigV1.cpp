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
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, width)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, height)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, orgX)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, orgY)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, binX)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, binY)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, exposureTime)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, coolingTemp)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, gainIndex)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, readoutSpeedIndex)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, exposureEventCode)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, numDelayShots)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, size)
  FUN0_WRAPPER(pypdsdata::Fli::ConfigV1, frameSize)

  PyMethodDef methods[] = {
    { "width",             width,             METH_NOARGS, "self.width() -> int\n\nReturns frame width" },
    { "height",            height,            METH_NOARGS, "self.height() -> int\n\nReturns fram height" },
    { "orgX",              orgX,              METH_NOARGS, "self.orgX() -> int\n\nReturns origin in X" },
    { "orgY",              orgY,              METH_NOARGS, "self.orgY() -> int\n\nReturns origin in Y" },
    { "binX",              binX,              METH_NOARGS, "self.binX() -> int\n\nReturns binning in X" },
    { "binY",              binY,              METH_NOARGS, "self.binY() -> int\n\nReturns binning in Y" },
    { "exposureTime",      exposureTime,      METH_NOARGS, "self.exposureTime() -> float\n\nReturns exposure time" },
    { "coolingTemp",       coolingTemp,       METH_NOARGS, "self.coolingTemp() -> float\n\nReturns integer number" },
    { "gainIndex",         gainIndex,         METH_NOARGS, "self.gainIndex() -> int\n\nReturns integer number" },
    { "readoutSpeedIndex", readoutSpeedIndex, METH_NOARGS, "self.readoutSpeedIndex() -> int\n\nReturns integer number" },
    { "exposureEventCode", exposureEventCode, METH_NOARGS, "self.exposureEventCode() -> int\n\nReturns integer number" },
    { "numDelayShots",     numDelayShots,     METH_NOARGS, "self.numDelayShots() -> int\n\nReturns integer number" },
    { "size",              size,              METH_NOARGS, "self.size() -> int\n\nReturns size of this object" },
    { "frameSize",         frameSize,         METH_NOARGS, "self.frameSize() -> int\n\nCalculate the frame size based on the current ROI and binning settings" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Fli::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Fli::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Fli::ConfigV1::print(std::ostream& str) const
{
  str << "Fli.ConfigV1(width=" << m_obj->width()
      << ", height=" << m_obj->height()
      << ", orgX=" << m_obj->orgX()
      << ", orgY=" << m_obj->orgY()
      << ", binX=" << m_obj->binX()
      << ", binY=" << m_obj->binY()
      << ", exposureEventCode=" << m_obj->exposureEventCode()
      << ", numDelayShots=" << m_obj->numDelayShots()
      << ", ...)";
}
