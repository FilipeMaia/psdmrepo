#ifndef H5DATATYPES_BLDDATAEBEAM_H
#define H5DATATYPES_BLDDATAEBEAM_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataEBeam.
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

struct BldDataEBeam_Data  {
  uint32_t    uDamageMask;
  double      fEbeamCharge;    /* in nC */
  double      fEbeamL3Energy;  /* in MeV */
  double      fEbeamLTUPosX;   /* in mm */
  double      fEbeamLTUPosY;   /* in mm */
  double      fEbeamLTUAngX;   /* in mrad */
  double      fEbeamLTUAngY;   /* in mrad */
};

class BldDataEBeam  {
public:

  typedef Pds::BldDataEBeam XtcType ;

  BldDataEBeam () {}
  BldDataEBeam ( const XtcType& xtc ) ;

  ~BldDataEBeam () ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

private:
  BldDataEBeam_Data m_data ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_BLDDATAEBEAM_H
