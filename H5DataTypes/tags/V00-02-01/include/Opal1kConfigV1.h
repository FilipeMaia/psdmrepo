#ifndef H5DATATYPES_OPAL1KCONFIGV1_H
#define H5DATATYPES_OPAL1KCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Opal1kConfigV1.
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
#include "pdsdata/opal1k/ConfigV1.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace H5DataTypes {

//
// Helper type for Pds::Opal1k::ConfigV1
//
class Opal1kConfigV1 {
public:

  typedef Pds::Opal1k::ConfigV1 XtcType ;

  Opal1kConfigV1() {}
  Opal1kConfigV1 ( const Pds::Opal1k::ConfigV1& config ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const Pds::Opal1k::ConfigV1& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:

  uint16_t black_level ;
  uint16_t gain_percent ;
  uint16_t output_offset ;
  uint8_t output_resolution ;
  uint8_t output_resolution_bits ;
  uint8_t vertical_binning ;
  uint8_t output_mirroring ;
  uint8_t vertical_remapping ;
  uint8_t defect_pixel_correction_enabled ;
  uint8_t output_lookup_table_enabled ;
  uint32_t number_of_defect_pixels ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_OPAL1KCONFIGV1_H
