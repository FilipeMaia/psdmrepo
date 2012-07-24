//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigV3...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "ConfigV3.h"

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
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, width)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, height)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, orgX)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, orgY)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, binX)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, binY)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, exposureTime)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, coolingTemp)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, gainIndex)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, readoutSpeedIndex)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, exposureEventCode)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, numDelayShots)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, size)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV3, frameSize)
  PyObject* _repr( PyObject *self );

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

  char typedoc[] = "Python class wrapping C++ Pds::Princeton::ConfigV3 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Princeton::ConfigV3::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "ConfigV3", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::Princeton::ConfigV3* obj = pypdsdata::Princeton::ConfigV3::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "princeton.ConfigV3(width=" << obj->width()
      << ", height=" << obj->height()
      << ", orgX=" << obj->orgX()
      << ", orgY=" << obj->orgY()
      << ", binX=" << obj->binX()
      << ", binY=" << obj->binY()
      << ", exposureEventCode=" << obj->exposureEventCode()
      << ", numDelayShots=" << obj->numDelayShots()
      << ", ...)";

  return PyString_FromString( str.str().c_str() );
}

}
