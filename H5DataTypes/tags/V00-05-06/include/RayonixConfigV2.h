#ifndef H5DATATYPES_RAYONIXCONFIGV2_H
#define H5DATATYPES_RAYONIXCONFIGV2_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RayonixConfigV2.
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

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/psddl/rayonix.ddl.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Rayonix::ConfigV2
//
class RayonixConfigV2  {
public:

  typedef Pds::Rayonix::ConfigV2 XtcType ;

  RayonixConfigV2 () {}
  RayonixConfigV2 ( const XtcType& data ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  // store single config object at specified location
  static void store( const XtcType& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return sizeof(xtc) ; }

private:

  uint8_t binning_f;
  uint8_t binning_s;
  int16_t testPattern;
  uint32_t exposure;
  uint32_t trigger;
  uint16_t rawMode;
  uint16_t darkFlag;
  uint32_t readoutMode;
  char deviceID[40];

};

} // namespace H5DataTypes

#endif // H5DATATYPES_RAYONIXCONFIGV2_H
