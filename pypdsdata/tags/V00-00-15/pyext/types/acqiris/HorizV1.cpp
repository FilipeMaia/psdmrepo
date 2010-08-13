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

  BaseType::initType( "HorizV1", module );
}
