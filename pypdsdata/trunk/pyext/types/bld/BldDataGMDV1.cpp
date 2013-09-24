//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataGMDV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataGMDV1.h"

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
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV1, milliJoulesPerPulse)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV1, milliJoulesAverage)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV1, correctedSumPerPulse)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV1, bgValuePerSample)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV1, relativeEnergyPerPulse)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"fMilliJoulesPerPulse",     milliJoulesPerPulse,      0, "Shot to shot pulse energy (mJ)", 0},
    {"fMilliJoulesAverage",      milliJoulesAverage,       0, "Average pulse energy from ION cup current (mJ)", 0},
    {"fCorrectedSumPerPulse",    correctedSumPerPulse,     0, "Bg corrected waveform integrated within limits in raw A/D counts", 0},
    {"fBgValuePerSample",        bgValuePerSample,         0, "Avg background value per sample in raw A/D counts", 0},
    {"fRelativeEnergyPerPulse",  relativeEnergyPerPulse,   0, "Shot by shot pulse energy in arbitrary units", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataGMDV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataGMDV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataGMDV1", module );
}

void
pypdsdata::Bld::BldDataGMDV1::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(fMilliJoulesPerPulse=" << m_obj->milliJoulesPerPulse()
        << ", fMilliJoulesAverage=" << m_obj->milliJoulesAverage()
        << ", fCorrectedSumPerPulse=" << m_obj->correctedSumPerPulse()
        << ", fBgValuePerSample=" << m_obj->bgValuePerSample()
        << ", fRelativeEnergyPerPulse=" << m_obj->relativeEnergyPerPulse()
        << ")";
  }
}
