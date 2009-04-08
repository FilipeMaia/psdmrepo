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

/**
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace H5DataTypes {

struct Opal1kConfigV1_Data {
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

class Opal1kConfigV1 {
public:
  Opal1kConfigV1() {}
  Opal1kConfigV1 ( const Pds::Opal1k::ConfigV1& config ) ;

  static hdf5pp::Type persType() ;

private:
  Opal1kConfigV1_Data m_data ;
};

void storeOpal1kConfigV1 ( const Pds::Opal1k::ConfigV1& config, hdf5pp::Group location ) ;

} // namespace H5DataTypes

#endif // H5DATATYPES_OPAL1KCONFIGV1_H
