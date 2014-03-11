#ifndef H5DATATYPES_BLDDATAEBEAMV5_H
#define H5DATATYPES_BLDDATAEBEAMV5_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeamV5.
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
#include "pdsdata/psddl/bld.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::BldDataEBeamV5
//
class BldDataEBeamV5  {
public:

  typedef Pds::Bld::BldDataEBeamV5 XtcType ;

  BldDataEBeamV5 () {}
  BldDataEBeamV5 ( const XtcType& xtc ) ;

  ~BldDataEBeamV5 () ;

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
  double      fEbeamUndPosX; /**< Und beam x-position in mm. */
  double      fEbeamUndPosY; /**< Und beam y-position in mm. */
  double      fEbeamUndAngX; /**< Und beam x-angle in mrad. */
  double      fEbeamUndAngY; /**< Und beam y-angle in mrad. */
  double      fEbeamXTCAVAmpl; /**< XTCAV Amplitude in MVolt. */
  double      fEbeamXTCAVPhase; /**< XTCAV Phase in degrees. */
  double      fEbeamDumpCharge; /**< Bunch charge at Dump in num. electrons */

};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAEBEAMV5_H
