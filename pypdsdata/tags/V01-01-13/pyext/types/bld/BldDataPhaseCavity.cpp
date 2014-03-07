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
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataPhaseCavity, fitTime1)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataPhaseCavity, fitTime2)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataPhaseCavity, charge1)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataPhaseCavity, charge2)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"fFitTime1", fitTime1, 0, "floating number, PV name: UND:R02:IOC:16:BAT:FitTime1, in pico-seconds", 0},
    {"fFitTime2", fitTime2, 0, "floating number, PV name: UND:R02:IOC:16:BAT:FitTime2, in pico-seconds", 0},
    {"fCharge1",  charge1,  0, "floating number, PV name: UND:R02:IOC:16:BAT:Charge1, in pico-columbs", 0},
    {"fCharge2",  charge2,  0, "floating number, PV name: UND:R02:IOC:16:BAT:Charge2, in pico-columbs", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataPhaseCavity class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataPhaseCavity::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataPhaseCavity", module );
}

void
pypdsdata::Bld::BldDataPhaseCavity::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(ft1=" << m_obj->fitTime1()
        << ", ft2=" << m_obj->fitTime2()
        << ", ch1=" << m_obj->charge1()
        << ", ch2=" << m_obj->charge2()
        << ")";
  }
}
