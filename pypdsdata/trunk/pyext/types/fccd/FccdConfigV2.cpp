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
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, size)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, ccdEnable)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, focusMode)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, exposureTime)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage1)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage2)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage3)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage4)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage5)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage6)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage7)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage8)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage9)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage10)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage11)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage12)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage13)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage14)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage15)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage16)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, dacVoltage17)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform0)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform1)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform2)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform3)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform4)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform5)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform6)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform7)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform8)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform9)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform10)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform11)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform12)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform13)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV2, waveform14)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    { "width",         width,         METH_NOARGS, "" },
    { "height",        height,        METH_NOARGS, "" },
    { "trimmedWidth",  trimmedWidth,  METH_NOARGS, "" },
    { "trimmedHeight", trimmedHeight, METH_NOARGS, "" },
    { "outputMode",    outputMode,    METH_NOARGS, "" },
    { "size",          size,          METH_NOARGS, "" },
    { "ccdEnable",     ccdEnable,     METH_NOARGS, "" },
    { "focusMode",     focusMode,     METH_NOARGS, "" },
    { "exposureTime",  exposureTime,  METH_NOARGS, "" },
    { "dacVoltage1",   dacVoltage1,   METH_NOARGS, "" },
    { "dacVoltage2",   dacVoltage2,   METH_NOARGS, "" },
    { "dacVoltage3",   dacVoltage3,   METH_NOARGS, "" },
    { "dacVoltage4",   dacVoltage4,   METH_NOARGS, "" },
    { "dacVoltage5",   dacVoltage5,   METH_NOARGS, "" },
    { "dacVoltage6",   dacVoltage6,   METH_NOARGS, "" },
    { "dacVoltage7",   dacVoltage7,   METH_NOARGS, "" },
    { "dacVoltage8",   dacVoltage8,   METH_NOARGS, "" },
    { "dacVoltage9",   dacVoltage9,   METH_NOARGS, "" },
    { "dacVoltage10",  dacVoltage10,  METH_NOARGS, "" },
    { "dacVoltage11",  dacVoltage11,  METH_NOARGS, "" },
    { "dacVoltage12",  dacVoltage12,  METH_NOARGS, "" },
    { "dacVoltage13",  dacVoltage13,  METH_NOARGS, "" },
    { "dacVoltage14",  dacVoltage14,  METH_NOARGS, "" },
    { "dacVoltage15",  dacVoltage15,  METH_NOARGS, "" },
    { "dacVoltage16",  dacVoltage16,  METH_NOARGS, "" },
    { "dacVoltage17",  dacVoltage17,  METH_NOARGS, "" },
    { "waveform0",     waveform0,     METH_NOARGS, "" },
    { "waveform1",     waveform1,     METH_NOARGS, "" },
    { "waveform2",     waveform2,     METH_NOARGS, "" },
    { "waveform3",     waveform3,     METH_NOARGS, "" },
    { "waveform4",     waveform4,     METH_NOARGS, "" },
    { "waveform5",     waveform5,     METH_NOARGS, "" },
    { "waveform6",     waveform6,     METH_NOARGS, "" },
    { "waveform7",     waveform7,     METH_NOARGS, "" },
    { "waveform8",     waveform8,     METH_NOARGS, "" },
    { "waveform9",     waveform9,     METH_NOARGS, "" },
    { "waveform10",    waveform10,    METH_NOARGS, "" },
    { "waveform11",    waveform11,    METH_NOARGS, "" },
    { "waveform12",    waveform12,    METH_NOARGS, "" },
    { "waveform13",    waveform13,    METH_NOARGS, "" },
    { "waveform14",    waveform14,    METH_NOARGS, "" },
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Depth", depthEnum.type() );
  PyDict_SetItemString( tp_dict, "Output_Source", outputSourceEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "FccdConfigV2", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::FCCD::FccdConfigV2* obj = pypdsdata::FCCD::FccdConfigV2::pdsObject(self);
  if (not obj) return 0;

  std::ostringstream str;
  str << "fccd.FccdConfigV2(outputMode" << obj->outputMode()
      << ", ccdEnable=" << (obj->ccdEnable() ? 1 : 0)
      << ", focusMode=" << (obj->focusMode() ? 1 : 0)
      << ", exposureTime=" << obj->exposureTime()
      << ", ...)";
  
  return PyString_FromString( str.str().c_str() );
}

}
