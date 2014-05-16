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
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, gasType)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, pressure)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, temperature)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, current)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, hvMeshElectron)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, hvMeshIon)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, hvMultIon)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, chargeQ)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, photonEnergy)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, multPulseIntensity)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, keithleyPulseIntensity)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, pulseEnergy)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, pulseEnergyFEE)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, transmission)
  MEMBER_WRAPPER_FROM_METHOD(pypdsdata::Bld::BldDataGMDV0, transmissionFEE)

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"strGasType",      gasType,       0, "string, gas type", 0},
    {"fPressure",       pressure,        0, "Pressure from Spinning Rotor Gauge", 0},
    {"fTemperature",    temperature,     0, "Temp from PT100", 0},
    {"fCurrent",        current,         0, "Current from Keithley Electrometer", 0},
    {"fHvMeshElectron", hvMeshElectron,  0, "HV Mesh Electron", 0},
    {"fHvMeshIon",      hvMeshIon,       0, "HV Mesh Ion", 0},
    {"fHvMultIon",      hvMultIon,       0, "HV Mult Ion", 0},
    {"fChargeQ",        chargeQ,         0, "Charge Q", 0},
    {"fPhotonEnergy",   photonEnergy,    0, "Photon Energy", 0},
    {"fMultPulseIntensity",multPulseIntensity, 0, "Pulse Intensity derived from Electron Multiplier", 0},
    {"fKeithleyPulseIntensity", keithleyPulseIntensity, 0, "Pulse Intensity derived from ION cup current", 0},
    {"fPulseEnergy",    pulseEnergy,     0, "Pulse Energy derived from Electron Multiplier", 0},
    {"fPulseEnergyFEE", pulseEnergyFEE,  0, "Pulse Energy from FEE Gas Detector", 0},
    {"fTransmission",   transmission,    0, "Transmission derived from Electron Multiplier", 0},
    {"fTransmissionFEE", transmissionFEE, 0, "Transmission from FEE Gas Detector", 0},
    {0, 0, 0, 0, 0}
  };

  char typedoc[] = "Python class wrapping C++ Pds::Bld::BldDataGMDV0 class.";
}

//              ----------------------------------------
//              -- Public Function Member Definitions --
//              ----------------------------------------

void
pypdsdata::Bld::BldDataGMDV0::initType( PyObject* module )
{
  PyTypeObject* type = BaseType::typeObject() ;
  type->tp_doc = ::typedoc;
  type->tp_getset = ::getset;

  BaseType::initType( "BldDataGMDV0", module );
}

void
pypdsdata::Bld::BldDataGMDV0::print(std::ostream& out) const
{
  if(not m_obj) {
    out << typeName() << "(None)";
  } else {
    out << typeName() << "(strGasType=" << m_obj->gasType()
        << ", fPressure=" << m_obj->pressure()
        << ", fTemperature=" << m_obj->temperature()
        << ", fCurrent=" << m_obj->current()
        << ", ...)";
  }
}
