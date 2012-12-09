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
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV1, fMilliJoulesPerPulse)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV1, fMilliJoulesAverage)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV1, fCorrectedSumPerPulse)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV1, fBgValuePerSample)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV1, fRelativeEnergyPerPulse)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV1, fSpare1)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"fMilliJoulesPerPulse",     fMilliJoulesPerPulse,      0, "Shot to shot pulse energy (mJ)", 0},
    {"fMilliJoulesAverage",      fMilliJoulesAverage,       0, "Average pulse energy from ION cup current (mJ)", 0},
    {"fCorrectedSumPerPulse",    fCorrectedSumPerPulse,     0, "Bg corrected waveform integrated within limits in raw A/D counts", 0},
    {"fBgValuePerSample",        fBgValuePerSample,         0, "Avg background value per sample in raw A/D counts", 0},
    {"fRelativeEnergyPerPulse",  fRelativeEnergyPerPulse,   0, "Shot by shot pulse energy in arbitrary units", 0},
    {"fSpare1",                  fSpare1,                   0, "Spare value for use as needed", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataGMDV1 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataGMDV1::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataGMDV1", module );
}

void
pypdsdata::BldDataGMDV1::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(fMilliJoulesPerPulse=" << m_obj->fMilliJoulesPerPulse
        << ", fMilliJoulesAverage=" << m_obj->fMilliJoulesAverage
        << ", fCorrectedSumPerPulse=" << m_obj->fCorrectedSumPerPulse
        << ", fBgValuePerSample=" << m_obj->fBgValuePerSample
        << ", fRelativeEnergyPerPulse=" << m_obj->fRelativeEnergyPerPulse
        << ")";
  }
}
