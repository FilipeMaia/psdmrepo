//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FccdConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "FccdConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "../../Exception.h"
#include "../../EnumType.h"
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------
namespace {

  pypdsdata::EnumType::Enum depthEnumValues[] = {
      { "Eight_bit",   Pds::FCCD::FccdConfigV2::Eight_bit },
      { "Sixteen_bit", Pds::FCCD::FccdConfigV2::Sixteen_bit },
      { 0, 0 }
  };
  pypdsdata::EnumType depthEnum ( "Depth", depthEnumValues );

  pypdsdata::EnumType::Enum outputSourceEnumValues[] = {
      { "Output_FIFO",   Pds::FCCD::FccdConfigV2::Output_FIFO },
      { "Test_Pattern1", Pds::FCCD::FccdConfigV2::Test_Pattern1 },
      { "Test_Pattern2", Pds::FCCD::FccdConfigV2::Test_Pattern2 },
      { "Test_Pattern3", Pds::FCCD::FccdConfigV2::Test_Pattern3 },
      { "Test_Pattern4", Pds::FCCD::FccdConfigV2::Test_Pattern4 },
      { 0, 0 }
  };
  pypdsdata::EnumType outputSourceEnum ( "Output_Source", outputSourceEnumValues );

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, width)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, height)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, trimmedWidth)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, trimmedHeight)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, outputMode)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, ccdEnable)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, focusMode)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, exposureTime)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltages)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveforms)

  PyMethodDef methods[] = {
    { "width",         width,         METH_NOARGS, "self.width() -> int\n\nReturns image width" },
    { "height",        height,        METH_NOARGS, "self.height() -> int\n\nReturns image height" },
    { "trimmedWidth",  trimmedWidth,  METH_NOARGS, "self.trimmedWidth() -> int\n\nReturns trimmed image width" },
    { "trimmedHeight", trimmedHeight, METH_NOARGS, "self.trimmedHeight() -> int\n\nReturns trimmed image height" },
    { "outputMode",    outputMode,    METH_NOARGS, "self.outputMode() -> int\n\nReturns integer number" },
    { "ccdEnable",     ccdEnable,     METH_NOARGS, "self.ccdEnable() -> bool\n\nReturns boolean" },
    { "focusMode",     focusMode,     METH_NOARGS, "self.focusMode() -> bool\n\nReturns boolean" },
    { "exposureTime",  exposureTime,  METH_NOARGS, "self.exposureTime() -> int\n\nReturns integer number" },
    { "dacVoltages",   dacVoltages,   METH_NOARGS, "self.dacVoltages() -> list of float\n\nReturns list of voltages" },
    { "waveforms",     waveforms,     METH_NOARGS, "self.waveforms() -> list of int\n\nReturns list of integer numbers" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::FCCD::FccdConfigV2 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::FCCD::FccdConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Depth", depthEnum.type() );
  PyDict_SetItemString( tp_dict, "Output_Source", outputSourceEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "FccdConfigV2", module );
}

void
pypdsdata::FCCD::FccdConfigV2::print(std::ostream& str) const
{
  str << "fccd.FccdConfigV2(outputMode" << m_obj->outputMode()
      << ", ccdEnable=" << (m_obj->ccdEnable() ? 1 : 0)
      << ", focusMode=" << (m_obj->focusMode() ? 1 : 0)
      << ", exposureTime=" << m_obj->exposureTime()
      << ", ...)";
}
