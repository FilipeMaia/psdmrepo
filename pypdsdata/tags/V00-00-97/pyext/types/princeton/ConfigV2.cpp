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
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, width)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, height)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, orgX)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, orgY)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, binX)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, binY)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, exposureTime)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, coolingTemp)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, gainIndex)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, readoutSpeedIndex)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, readoutEventCode)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, delayMode)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, size)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV2, frameSize)

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
    { "readoutEventCode",  readoutEventCode,  METH_NOARGS, "self.readoutEventCode() -> int\n\nReturns integer number" },
    { "delayMode",         delayMode,         METH_NOARGS, "self.delayMode() -> int\n\nReturns integer number" },
    { "size",              size,              METH_NOARGS, "self.size() -> int\n\nReturns size of this object" },
    { "frameSize",         frameSize,         METH_NOARGS, "self.frameSize() -> int\n\nCalculate the frame size based on the current ROI and binning settings" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Princeton::ConfigV2 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Princeton::ConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV2", module );
}

void
pypdsdata::Princeton::ConfigV2::print(std::ostream& out) const
{
  out << "princeton.ConfigV2(width=" << m_obj->width()
      << ", height=" << m_obj->height()
      << ", orgX=" << m_obj->orgX()
      << ", orgY=" << m_obj->orgY()
      << ", binX=" << m_obj->binX()
      << ", binY=" << m_obj->binY()
      << ", readoutEventCode=" << m_obj->readoutEventCode()
      << ", delayMode=" << m_obj->delayMode()
      << ", ...)";
}
