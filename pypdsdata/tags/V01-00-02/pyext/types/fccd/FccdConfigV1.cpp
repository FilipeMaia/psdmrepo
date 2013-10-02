//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class FccdConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "FccdConfigV1.h"

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
      { "Sixteen_bit", Pds::FCCD::FccdConfigV1::Sixteen_bit },
      { 0, 0 }
  };
  pypdsdata::EnumType depthEnum ( "Depth", depthEnumValues );

  pypdsdata::EnumType::Enum outputSourceEnumValues[] = {
      { "Output_FIFO",     Pds::FCCD::FccdConfigV1::Output_FIFO },
      { "Output_Pattern4", Pds::FCCD::FccdConfigV1::Output_Pattern4 },
      { 0, 0 }
  };
  pypdsdata::EnumType outputSourceEnum ( "Output_Source", outputSourceEnumValues );

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV1, width)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV1, height)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV1, trimmedWidth)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV1, trimmedHeight)
  FUN0_WRAPPER(pypdsdata::FCCD::FccdConfigV1, outputMode)

  PyMethodDef methods[] = {
    { "width",         width,         METH_NOARGS, "self.width() -> int\n\nReturns image width" },
    { "height",        height,        METH_NOARGS, "self.height() -> int\n\nReturns image height" },
    { "trimmedWidth",  trimmedWidth,  METH_NOARGS, "self.trimmedWidth() -> int\n\nReturns trimmed image width" },
    { "trimmedHeight", trimmedHeight, METH_NOARGS, "self.trimmedHeight() -> int\n\nReturns trimmed image height" },
    { "outputMode",    outputMode,    METH_NOARGS, "self.outputMode() -> int\n\nReturns integer number" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::FCCD::FccdConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::FCCD::FccdConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  PyObject* tp_dict = PyDict_New();
  PyDict_SetItemString( tp_dict, "Depth", depthEnum.type() );
  PyDict_SetItemString( tp_dict, "Output_Source", outputSourceEnum.type() );
  type->tp_dict = tp_dict;

  BaseType::initType( "FccdConfigV1", module );
}

void
pypdsdata::FCCD::FccdConfigV1::print(std::ostream& str) const
{
  str << "fccd.FccdConfigV1(outputMode" << m_obj->outputMode() << ")";
}
