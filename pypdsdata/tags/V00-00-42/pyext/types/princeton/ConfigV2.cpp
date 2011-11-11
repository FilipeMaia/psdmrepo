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
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "width",                  width,                  METH_NOARGS, "" },
    { "height",                 height,                 METH_NOARGS, "" },
    { "orgX",                   orgX,                   METH_NOARGS, "" },
    { "orgY",                   orgY,                   METH_NOARGS, "" },
    { "binX",                   binX,                   METH_NOARGS, "" },
    { "binY",                   binY,                   METH_NOARGS, "" },
    { "exposureTime",           exposureTime,           METH_NOARGS, "" },
    { "coolingTemp",            coolingTemp,            METH_NOARGS, "" },
    { "gainIndex",              gainIndex,              METH_NOARGS, "" },
    { "readoutSpeedIndex",      readoutSpeedIndex,      METH_NOARGS, "" },
    { "readoutEventCode",       readoutEventCode,       METH_NOARGS, "" },
    { "delayMode",              delayMode,              METH_NOARGS, "" },
    { "size",                   size,                   METH_NOARGS, "" },
    { "frameSize",              frameSize,              METH_NOARGS, "" },
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "ConfigV2", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::Princeton::ConfigV2* obj = pypdsdata::Princeton::ConfigV2::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "princeton.ConfigV2(width=" << obj->width()
      << ", height=" << obj->height()
      << ", readoutEventCode=" << obj->readoutEventCode()
      << ", delayMode=" << obj->delayMode()
      << ", ...)";

  return PyString_FromString( str.str().c_str() );
}

}
