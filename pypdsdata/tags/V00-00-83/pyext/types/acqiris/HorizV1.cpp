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

  PyMethodDef methods[] = {
    {"sampInterval", sampInterval, METH_NOARGS,  "self.sampInterval() -> float\n\nReturns floating number" },
    {"delayTime",    delayTime,    METH_NOARGS,  "self.delayTime() -> float\n\nReturns floating number" },
    {"nbrSamples",   nbrSamples,   METH_NOARGS,  "self.nbrSamples() -> int\n\nReturns number of samples" },
    {"nbrSegments",  nbrSegments,  METH_NOARGS,  "self.nbrSegments() -> int\n\nReturns number of segments" },
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

void 
pypdsdata::Acqiris::HorizV1::print(std::ostream& out) const
{
  if(not m_obj) {
    out << "acqiris.HorizV1(None)";
  } else {  
    out << "acqiris.HorizV1(sampInterval=" << m_obj->sampInterval()
        << ", delayTime=" << m_obj->delayTime()
        << ", nbrSamples=" << m_obj->nbrSamples()
        << ", nbrSegments=" << m_obj->nbrSegments()
        << ", ...)" ;
  }
}
