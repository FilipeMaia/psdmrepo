#ifndef H5DATATYPES_BLDDATAEBEAMV2_H
#define H5DATATYPES_BLDDATAEBEAMV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV2.
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

class BldDataEBeamV2  {
public:

  typedef Pds::BldDataEBeamV2 XtcType ;

  BldDataEBeamV2 () {}
  BldDataEBeamV2 ( const XtcType& xtc ) ;

  ~BldDataEBeamV2 () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:

  uint32_t    uDamageMask;
  double      fEbeamCharge;    /* in nC */
  double      fEbeamL3Energy;  /* in MeV */
  double      fEbeamLTUPosX;   /* in mm */
  double      fEbeamLTUPosY;   /* in mm */
  double      fEbeamLTUAngX;   /* in mrad */
  double      fEbeamLTUAngY;   /* in mrad */
  double      fEbeamPkCurrBC2; /* in Amps */
  double      fEbeamEnergyBC2; /* in MeV */

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAEBEAMV2_H
