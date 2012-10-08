#ifndef H5DATATYPES_BLDDATAGMDV0_H
#define H5DATATYPES_BLDDATAGMDV0_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataGMDV0.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Type.h"
#include "pdsdata/bld/bldData.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

class BldDataGMDV0  {
public:

  typedef Pds::BldDataGMDV0 XtcType ;

  BldDataGMDV0 () {}
  BldDataGMDV0 ( const XtcType& xtc ) ;

  ~BldDataGMDV0 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:

  char     strGasType[32];  // Gas Type
  double   fPressure;       // Pressure from Spinning Rotor Gauge, SXR:GMD:SRG:01:Pressure
  double   fTemperature;    // Temp from PT100, SXR:GMD:RTD:40:RAW_AI
  double   fCurrent;        // Current from Keithley Electrometer, SXR:GMD:ETM:01:Reading
  int32_t  iHvMeshElectron; // HV Mesh Electron, SXR:GMD:VHQ1:ChA:VoltageMeasure
  int32_t  iHvMeshIon;      // HV Mesh Ion,      SXR:GMD:VHQ1:ChB:VoltageMeasure
  int32_t  iHvMultIon;      // HV Mult Ion,      SXR:GMD:VHQ1:ChB:VoltageMeasure
  double   fChargeQ;        // Charge Q, SXR:GMD:IMD:Charge_Q
  double   fPhotonEnergy;   // Photon Energy, SIOC:SYS0:ML00:AO627
  double   fPhotonsPerPulse;// Photons Per Pulse, SXR:GMD:IMD:CalcIMD:PhotonsPerPulse
  double   fSpare1;         // Spare 1
  double   fSpare2;         // Spare 2
  double   fSpare3;         // Spare 3
  double   fSpare4;         // Spare 4
  double   fSpare5;         // Spare 5
  double   fSpare6;         // Spare 6

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAGMDV0_H
