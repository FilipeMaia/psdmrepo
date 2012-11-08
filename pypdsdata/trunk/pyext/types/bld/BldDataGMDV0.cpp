//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataGMDV0...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "BldDataGMDV0.h"

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
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, strGasType)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fPressure)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fTemperature)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fCurrent)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fHvMeshElectron)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fHvMeshIon)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fHvMultIon)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fChargeQ)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fPhotonEnergy)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fMultPulseIntensity)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fKeithleyPulseIntensity)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fPulseEnergy)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fPulseEnergyFEE)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fTransmission)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fTransmissionFEE)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fSpare6)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"strGasType",      strGasType,       0, "string, gas type", 0},
    {"fPressure",       fPressure,        0, "Pressure from Spinning Rotor Gauge", 0},
    {"fTemperature",    fTemperature,     0, "Temp from PT100", 0},
    {"fCurrent",        fCurrent,         0, "Current from Keithley Electrometer", 0},
    {"fHvMeshElectron", fHvMeshElectron,  0, "HV Mesh Electron", 0},
    {"fHvMeshIon",      fHvMeshIon,       0, "HV Mesh Ion", 0},
    {"fHvMultIon",      fHvMultIon,       0, "HV Mult Ion", 0},
    {"fChargeQ",        fChargeQ,         0, "Charge Q", 0},
    {"fPhotonEnergy",   fPhotonEnergy,    0, "Photon Energy", 0},
    {"fMultPulseIntensity",fMultPulseIntensity, 0, "Pulse Intensity derived from Electron Multiplier", 0},
    {"fKeithleyPulseIntensity", fKeithleyPulseIntensity, 0, "Pulse Intensity derived from ION cup current", 0},
    {"fPulseEnergy",    fPulseEnergy,     0, "Pulse Energy derived from Electron Multiplier", 0},
    {"fPulseEnergyFEE", fPulseEnergyFEE,  0, "Pulse Energy from FEE Gas Detector", 0},
    {"fTransmission",   fTransmission,    0, "Transmission derived from Electron Multiplier", 0},
    {"fTransmissionFEE", fTransmissionFEE, 0, "Transmission from FEE Gas Detector", 0},
    {"fSpare6",         fSpare6,          0, "Spare 6", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::BldDataGMDV0 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::BldDataGMDV0::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataGMDV0", module );
}

void
pypdsdata::BldDataGMDV0::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(strGasType=" << m_obj->strGasType
        << ", fPressure=" << m_obj->fPressure
        << ", fTemperature=" << m_obj->fTemperature
        << ", fCurrent=" << m_obj->fCurrent
        << ", ...)";
  }
}
