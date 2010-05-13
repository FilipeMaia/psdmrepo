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
#include "SITConfig/SITConfig.h"

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
#include "../../Exception.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, width)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, height)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, orgX)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, orgY)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, binX)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, binY)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, exposureTime)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, coolingTemp)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, readoutSpeedIndex)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, readoutEventCode)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, delayMode)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, size)
  FUN0_WRAPPER(pypdsdata::Princeton::ConfigV1, frameSize)

  PyMethodDef methods[] = {
    { "width",                  width,                  METH_NOARGS, "" },
    { "height",                 height,                 METH_NOARGS, "" },
    { "orgX",                   orgX,                   METH_NOARGS, "" },
    { "orgY",                   orgY,                   METH_NOARGS, "" },
    { "binX",                   binX,                   METH_NOARGS, "" },
    { "binY",                   binY,                   METH_NOARGS, "" },
    { "exposureTime",           exposureTime,           METH_NOARGS, "" },
    { "coolingTemp",            coolingTemp,            METH_NOARGS, "" },
    { "readoutSpeedIndex",      readoutSpeedIndex,      METH_NOARGS, "" },
    { "readoutEventCode",       readoutEventCode,       METH_NOARGS, "" },
    { "delayMode",              delayMode,              METH_NOARGS, "" },
    { "size",                   size,                   METH_NOARGS, "" },
    { "frameSize",              frameSize,              METH_NOARGS, "" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Princeton::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Princeton::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}
