//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPhaseCavity...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataPhaseCavity.h"

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
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fFitTime1)
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fFitTime2)
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fCharge1)
  MEMBER_WRAPPER(pypdsdata::BldDataPhaseCavity, fCharge2)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"fFitTime1", fFitTime1, 0, "floating number, PV name: UND:R02:IOC:16:BAT:FitTime1, in pico-seconds", 0},
    {"fFitTime2", fFitTime2, 0, "floating number, PV name: UND:R02:IOC:16:BAT:FitTime2, in pico-seconds", 0},
    {"fCharge1",  fCharge1,  0, "floating number, PV name: UND:R02:IOC:16:BAT:Charge1, in pico-columbs", 0},
    {"fCharge2",  fCharge2,  0, "floating number, PV name: UND:R02:IOC:16:BAT:Charge2, in pico-columbs", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataPhaseCavity class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataPhaseCavity::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataPhaseCavity", module );
}

void
pypdsdata::BldDataPhaseCavity::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(ft1=" << m_obj->fFitTime1
        << ", ft2=" << m_obj->fFitTime2
        << ", ch1=" << m_obj->fCharge1
        << ", ch2=" << m_obj->fCharge2
        << ")";
  }
}
