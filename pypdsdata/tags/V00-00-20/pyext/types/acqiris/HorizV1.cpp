//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Acqiris_HorizV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "HorizV1.h"

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

  // methods
  FUN0_WRAPPER(pypdsdata::Acqiris::HorizV1, sampInterval)
  FUN0_WRAPPER(pypdsdata::Acqiris::HorizV1, delayTime)
  FUN0_WRAPPER(pypdsdata::Acqiris::HorizV1, nbrSamples)
  FUN0_WRAPPER(pypdsdata::Acqiris::HorizV1, nbrSegments)
  PyObject* _repr( PyObject *self );

  PyMethodDef methods[] = {
    {"sampInterval", sampInterval, METH_NOARGS,  "Returns floating number" },
    {"delayTime",    delayTime,    METH_NOARGS,  "Returns floating number" },
    {"nbrSamples",   nbrSamples,   METH_NOARGS,  "Returns number of samples" },
    {"nbrSegments",  nbrSegments,  METH_NOARGS,  "Returns number of segments" },
    {0, 0, 0, 0}
   };

  char typedoc[] = "Python class wrapping C++ Pds::Acqiris::HorizV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Acqiris::HorizV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_methods = ::methods;
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "HorizV1", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::Acqiris::HorizV1* obj = pypdsdata::Acqiris::HorizV1::pdsObject(self);
  if(not obj) return 0;

  std::ostringstream str;
  str << "acqiris.HorizV1(sampInterval=" << obj->sampInterval()
      << ", delayTime=" << obj->delayTime()
      << ", nbrSamples=" << obj->nbrSamples()
      << ", nbrSegments=" << obj->nbrSegments()
      << ", ...)" ;
  return PyString_FromString( str.str().c_str() );
}

}
