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
  PyObject* _repr( PyObject *self );

  // methods
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV1, exposureTime)
  FUN0_WRAPPER(pypdsdata::OceanOptics::ConfigV1, strayLightConstant)
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
  type->tp_str = _repr ;
  type->tp_repr = _repr ;

  BaseType::initType( "ConfigV1", module );
}


namespace {

PyObject*
waveLenCalib( PyObject* self, PyObject* )
{
  Pds::OceanOptics::ConfigV1* obj = pypdsdata::OceanOptics::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  const unsigned size = 4;
  PyObject* list = PyList_New(size);

  // copy PvConfig objects to the list
  for ( unsigned i = 0; i < size; ++ i ) {
    PyList_SET_ITEM(list, i, PyFloat_FromDouble(obj->waveLenCalib(i)));
  }

  return list;
}

PyObject*
nonlinCorrect( PyObject* self, PyObject* )
{
  Pds::OceanOptics::ConfigV1* obj = pypdsdata::OceanOptics::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  const unsigned size = 8;
  PyObject* list = PyList_New(size);

  // copy PvConfig objects to the list
  for ( unsigned i = 0; i < size; ++ i ) {
    PyList_SET_ITEM(list, i, PyFloat_FromDouble(obj->nonlinCorrect(i)));
  }

  return list;
}

PyObject*
_repr( PyObject *self )
{
  Pds::OceanOptics::ConfigV1* obj = pypdsdata::OceanOptics::ConfigV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "oceanoptics.ConfigV1(exposureTime=" << obj->exposureTime()
      << ", waveLenCalib=[";
  for ( unsigned i = 0; i < 4; ++ i ) {
    if (i) str << ",";
    str << obj->waveLenCalib(i);
  }
  str << "], strayLightConstant=" << obj->strayLightConstant()
      << ", nonlinCorrect=[";
  for ( unsigned i = 0; i < 8; ++ i ) {
    if (i) str << ",";
    str << obj->nonlinCorrect(i);
  }
  str << "])";
  
  return PyString_FromString(str.str().c_str());
}

}
