#ifndef H5DATATYPES_BLDDATAGMDV1_H
#define H5DATATYPES_BLDDATAGMDV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataGMDV1.
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

class BldDataGMDV1  {
public:

  typedef Pds::BldDataGMDV1 XtcType ;

  BldDataGMDV1 () {}
  BldDataGMDV1 ( const XtcType& xtc ) ;

  ~BldDataGMDV1 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:

  double  fMilliJoulesPerPulse;    // Shot to shot pulse energy (mJ)
  double  fMilliJoulesAverage;     // Average pulse energy from ION cup current (mJ)
  double  fCorrectedSumPerPulse;   // Bg corrected waveform integrated within limits in raw A/D counts
  double  fBgValuePerSample;       // Avg background value per sample in raw A/D counts
  double  fRelativeEnergyPerPulse; // Shot by shot pulse energy in arbitrary units
  double  fSpare1;                 // Spare value for use as needed

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAGMDV1_H
