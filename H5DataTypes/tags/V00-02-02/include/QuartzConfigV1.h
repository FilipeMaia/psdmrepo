#ifndef H5DATATYPES_QUARTZCONFIGV1_H
#define H5DATATYPES_QUARTZCONFIGV1_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class QuartzConfigV1.
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
#include "pdsdata/quartz/ConfigV1.hh"

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

class QuartzConfigV1 {
public:

  typedef Pds::Quartz::ConfigV1 XtcType ;

  QuartzConfigV1() {}
  QuartzConfigV1 ( const Pds::Quartz::ConfigV1& config ) ;

  static hdf5pp::Type stored_type() ;
  static hdf5pp::Type native_type() ;

  static void store ( const Pds::Quartz::ConfigV1& config, hdf5pp::Group location ) ;

  static size_t xtcSize( const XtcType& xtc ) { return xtc.size() ; }

private:

  uint16_t black_level ;
  uint16_t gain_percent ;
  uint16_t output_offset ;
  uint8_t output_resolution ;
  uint8_t output_resolution_bits ;
  uint8_t horizontal_binning ;
  uint8_t vertical_binning ;
  uint8_t output_mirroring ;
  uint8_t defect_pixel_correction_enabled ;
  uint8_t output_lookup_table_enabled ;
  uint32_t number_of_defect_pixels ;
};

} // namespace H5DataTypes

#endif // H5DATATYPES_QUARTZCONFIGV1_H
