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

  pypdsdata::EnumType::Enum readoutModeEnumValues[] = {
      { "Standard",     Pds::Rayonix::ConfigV1::Standard },
      { "HighGain",     Pds::Rayonix::ConfigV1::HighGain },
      { "LowNoise",     Pds::Rayonix::ConfigV1::LowNoise },
      { "EDR",          Pds::Rayonix::ConfigV1::EDR },
      { 0, 0 }
  };
  pypdsdata::EnumType readoutModeEnum ( "ReadoutMode", readoutModeEnumValues );

  // type-specific methods
  FUN0_WRAPPER(pypdsdata::Rayonix::ConfigV1, binning_f)
  FUN0_WRAPPER(pypdsdata::Rayonix::ConfigV1, binning_s)
  FUN0_WRAPPER(pypdsdata::Rayonix::ConfigV1, exposure)
  FUN0_WRAPPER(pypdsdata::Rayonix::ConfigV1, trigger)
  FUN0_WRAPPER(pypdsdata::Rayonix::ConfigV1, rawMode)
  FUN0_WRAPPER(pypdsdata::Rayonix::ConfigV1, darkFlag)
  ENUM_FUN0_WRAPPER(pypdsdata::Rayonix::ConfigV1, readoutMode, readoutModeEnum)
  FUN0_WRAPPER(pypdsdata::Rayonix::ConfigV1, deviceID)

  PyMethodDef methods[] = {
    { "binning_f",     binning_f,     METH_NOARGS, "self.binning_f() -> int\n\nReturns integer number" },
    { "binning_s",     binning_s,     METH_NOARGS, "self.binning_s() -> int\n\nReturns integer number" },
    { "exposure",      exposure,      METH_NOARGS, "self.exposure() -> int\n\nReturns integer number" },
    { "trigger",       trigger,       METH_NOARGS, "self.trigger() -> int\n\nReturns integer number" },
    { "rawMode",       rawMode,       METH_NOARGS, "self.rawMode() -> int\n\nReturns integer number" },
    { "darkFlag",      darkFlag,      METH_NOARGS, "self.darkFlag() -> int\n\nReturns integer number" },
    { "readoutMode",   readoutMode,   METH_NOARGS, "self.readoutMode() -> int\n\nReturns integer number" },
    { "deviceID",      deviceID,      METH_NOARGS, "self.deviceID() -> string\n\nReturns string" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Rayonix::ConfigV1 class.";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

void
pypdsdata::Rayonix::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  // define class attributes for enums
  type->tp_dict = PyDict_New();
  PyDict_SetItemString( type->tp_dict, "ReadoutMode", readoutModeEnum.type() );

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::Rayonix::ConfigV1::print(std::ostream& str) const
{
  str << "Rayonix.ConfigV1(binning_f=" << int(m_obj->binning_f())
      << ", binning_s=" << int(m_obj->binning_s())
      << ", exposure=" << m_obj->exposure()
      << ", trigger=" << m_obj->trigger()
      << ", rawMode=" << m_obj->rawMode()
      << ", darkFlag=" << m_obj->darkFlag()
      << ", readoutMode=" << m_obj->readoutMode()
      << ", deviceID=" << m_obj->deviceID()
      << ")" ;
}
