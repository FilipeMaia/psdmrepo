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
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV2, exposureTime)
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV2, deviceType)
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV2, waveLenCalib)
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV2, strayLightConstant)
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV2, nonlinCorrect)
  PyObject* waveLenCalib( PyObject* self, PyObject* );
  PyObject* nonlinCorrect( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    {"exposureTime",        exposureTime,       METH_NOARGS,  "self.exposureTime() -> float\n\nReturns floating number." },
    {"deviceType",          deviceType,         METH_NOARGS,  "self.deviceType() -> int\n\nReturns device type." },
    {"waveLenCalib",        waveLenCalib,       METH_NOARGS,
        "self.waveLenCalib() -> list of floats\n\nReturns list of 4 floating numbers. Wavelength Calibration Coefficient; 0th - 3rd order." },
    {"strayLightConstant",  strayLightConstant, METH_NOARGS,  "self.strayLightConstant() -> float\n\nReturns floating number." },
    {"nonlinCorrect",       nonlinCorrect,      METH_NOARGS,
        "self.nonlinCorrect() -> list of floats\n\nReturns list of 8 floating numbers. Non-linearity correction coefficient; 0th - 7th order." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::OceanOptics::ConfigV2 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::OceanOptics::ConfigV2::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV2", module );
}

void
pypdsdata::OceanOptics::ConfigV2::print(std::ostream& str) const
{
  str << "oceanoptics.ConfigV2(exposureTime=" << m_obj->exposureTime()
      << ", deviceType=" << m_obj->deviceType()
      << ", waveLenCalib=" << m_obj->waveLenCalib()
      << ", strayLightConstant=" << m_obj->strayLightConstant()
      << ", nonlinCorrect=" << m_obj->nonlinCorrect()
      << ")";
}
