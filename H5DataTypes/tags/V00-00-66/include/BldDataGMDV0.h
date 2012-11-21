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

  char    strGasType[32];         // Gas Type
  double  fPressure;              // Pressure from Spinning Rotor Gauge
  double  fTemperature;           // Temp from PT100
  double  fCurrent;               // Current from Keithley Electrometer
  double  fHvMeshElectron;        // HV Mesh Electron
  double  fHvMeshIon;             // HV Mesh Ion
  double  fHvMultIon;             // HV Mult Ion
  double  fChargeQ;               // Charge Q
  double  fPhotonEnergy;          // Photon Energy
  double  fMultPulseIntensity;    // Pulse Intensity derived from Electron Multiplier
  double  fKeithleyPulseIntensity;// Pulse Intensity derived from ION cup current
  double  fPulseEnergy;           // Pulse Energy derived from Electron Multiplier
  double  fPulseEnergyFEE;        // Pulse Energy from FEE Gas Detector
  double  fTransmission;          // Transmission derived from Electron Multiplier
  double  fTransmissionFEE;       // Transmission from FEE Gas Detector
  double  fSpare6;                // Spare 6

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAGMDV0_H
