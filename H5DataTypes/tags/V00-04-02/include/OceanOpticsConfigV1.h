#ifndef H5DATATYPES_OCEANOPTICSCONFIGV1_H
#define H5DATATYPES_OCEANOPTICSCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsConfigV1.
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
#include "hdf5pp/Group.h"
#include "pdsdata/oceanoptics/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::OceanOptics::ConfigV1
//
class OceanOpticsConfigV1  {
public:

  typedef Pds::OceanOptics::ConfigV1 XtcType ;

  OceanOpticsConfigV1 () {}
  OceanOpticsConfigV1 ( const XtcType& config ) ;

  ~OceanOpticsConfigV1();

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:

  float             exposureTime;
  double            waveLenCalib[4];
  double            nonlinCorrect[8];
  double            strayLightConstant;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_OCEANOPTICSCONFIGV1_H
