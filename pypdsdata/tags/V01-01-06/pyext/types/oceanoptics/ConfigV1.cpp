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
#include "../TypeLib.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // methods
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV1, exposureTime)
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV1, waveLenCalib)
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV1, strayLightConstant)
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV1, nonlinCorrect)
  PyObject* waveLenCalib( PyObject* self, PyObject* );
  PyObject* nonlinCorrect( PyObject* self, PyObject* );

  PyMethodDef methods[] = {
    {"exposureTime",        exposureTime,       METH_NOARGS,  "self.exposureTime() -> float\n\nReturns floating number." },
    {"waveLenCalib",        waveLenCalib,       METH_NOARGS,
        "self.waveLenCalib() -> list of floats\n\nReturns list of 4 floating numbers. Wavelength Calibration Coefficient; 0th - 3rd order." },
    {"strayLightConstant",  strayLightConstant, METH_NOARGS,  "self.strayLightConstant() -> float\n\nReturns floating number." },
    {"nonlinCorrect",       nonlinCorrect,      METH_NOARGS,
        "self.nonlinCorrect() -> list of floats\n\nReturns list of 8 floating numbers. Non-linearity correction coefficient; 0th - 7th order." },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::OceanOptics::ConfigV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::OceanOptics::ConfigV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;

  BaseType::initType( "ConfigV1", module );
}

void
pypdsdata::OceanOptics::ConfigV1::print(std::ostream& str) const
{
  str << "oceanoptics.ConfigV1(exposureTime=" << m_obj->exposureTime()
      << ", waveLenCalib=" << m_obj->waveLenCalib()
      << ", strayLightConstant=" << m_obj->strayLightConstant()
      << ", nonlinCorrect=" << m_obj->nonlinCorrect()
      << ")";
}
