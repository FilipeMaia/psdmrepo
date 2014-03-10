#ifndef H5DATATYPES_OCEANOPTICSCONFIGV2_H
#define H5DATATYPES_OCEANOPTICSCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsConfigV2.
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
#include "pdsdata/psddl/oceanoptics.ddl.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::OceanOptics::ConfigV2
//
class OceanOpticsConfigV2  {
public:

  typedef Pds::OceanOptics::ConfigV2 XtcType ;

  OceanOpticsConfigV2 () {}
  OceanOpticsConfigV2 ( const XtcType& config ) ;

  ~OceanOpticsConfigV2();

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof xtc ; }

private:

  float             exposureTime;
  int32_t           deviceType;
  double            waveLenCalib[4];
  double            nonlinCorrect[8];
  double            strayLightConstant;

};

} // namespace H5DataTypes

#endif // H5DATATYPES_OCEANOPTICSCONFIGV2_H
