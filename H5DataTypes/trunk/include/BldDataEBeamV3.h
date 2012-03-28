#ifndef H5DATATYPES_BldDataEBeamV3_H
#define H5DATATYPES_BldDataEBeamV3_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV3.
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

class BldDataEBeamV3  {
public:

  typedef Pds::BldDataEBeamV3 XtcType ;

  BldDataEBeamV3 () {}
  BldDataEBeamV3 ( const XtcType& xtc ) ;

  ~BldDataEBeamV3 () ;

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
  double      fEbeamEnergyBC2; /* in mm, beam position (related to energy) */
  double      fEbeamPkCurrBC1; /* in Amps */
  double      fEbeamEnergyBC1; /* in mm, beam position (related to energy) */

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BldDataEBeamV3_H
