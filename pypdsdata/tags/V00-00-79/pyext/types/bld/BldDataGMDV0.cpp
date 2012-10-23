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
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, iHvMeshElectron)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, iHvMeshIon)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, iHvMultIon)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fChargeQ)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fPhotonEnergy)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fPhotonsPerPulse)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fSpare1)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fSpare2)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fSpare3)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fSpare4)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fSpare5)
  MEMBER_WRAPPER(pypdsdata::BldDataGMDV0, fSpare6)
  PyObject* _repr( PyObject *self );

  // disable warnings for non-const strings, this is a temporary measure
  // newer Python versions should get constness correctly
#pragma GCC diagnostic ignored "-Wwrite-strings"
  PyGetSetDef getset[] = {
    {"strGasType",      strGasType,       0, "string, gas type", 0},
    {"fPressure",       fPressure,        0, "Pressure from Spinning Rotor Gauge, SXR:GMD:SRG:01:Pressure", 0},
    {"fTemperature",    fTemperature,     0, "Temp from PT100, SXR:GMD:RTD:40:RAW_AI", 0},
    {"fCurrent",        fCurrent,         0, "Current from Keithley Electrometer, SXR:GMD:ETM:01:Reading", 0},
    {"iHvMeshElectron", iHvMeshElectron,  0, "HV Mesh Electron, SXR:GMD:VHQ1:ChA:VoltageMeasure", 0},
    {"iHvMeshIon",      iHvMeshIon,       0, "HV Mesh Ion, SXR:GMD:VHQ1:ChB:VoltageMeasure", 0},
    {"iHvMultIon",      iHvMultIon,       0, "HV Mult Ion, SXR:GMD:VHQ1:ChB:VoltageMeasure", 0},
    {"fChargeQ",        fChargeQ,         0, "Charge Q, SXR:GMD:IMD:Charge_Q", 0},
    {"fPhotonEnergy",   fPhotonEnergy,    0, "Photon Energy, SIOC:SYS0:ML00:AO627", 0},
    {"fPhotonsPerPulse",fPhotonsPerPulse, 0, "Photons Per Pulse, SXR:GMD:IMD:CalcIMD:PhotonsPerPulse", 0},
    {"fSpare1",         fSpare1,          0, "Spare 1", 0},
    {"fSpare2",         fSpare2,          0, "Spare 2", 0},
    {"fSpare3",         fSpare3,          0, "Spare 3", 0},
    {"fSpare4",         fSpare4,          0, "Spare 4", 0},
    {"fSpare5",         fSpare5,          0, "Spare 5", 0},
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
  type->tp_str = _repr;
  type->tp_repr = _repr;

  BaseType::initType( "BldDataGMDV0", module );
}

namespace {

PyObject*
_repr( PyObject *self )
{
  Pds::BldDataGMDV0* pdsObj = pypdsdata::BldDataGMDV0::pdsObject(self);
  if(not pdsObj) return 0;

  char buf[196];
  snprintf( buf, sizeof buf, "BldDataGMDV0(strGasType=%s, fPressure=%g, fTemperature=%g, fCurrent=%g, ...)",
            pdsObj->strGasType, pdsObj->fPressure, pdsObj->fTemperature, pdsObj->fCurrent );
  return PyString_FromString( buf );
}

}
