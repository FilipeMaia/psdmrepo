//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PulnixTM6740ConfigV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PulnixTM6740ConfigV2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

//----------------
// Constructors --
//----------------
PulnixTM6740ConfigV2::PulnixTM6740ConfigV2 ( const Pds::Pulnix::TM6740ConfigV2& config )
{
}

hdf5pp::Type
PulnixTM6740ConfigV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PulnixTM6740ConfigV2::native_type()
{
  hdf5pp::EnumType<uint8_t> depthEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  depthEnum.insert ( "Eight_bit", Pds::Pulnix::TM6740ConfigV2::Eight_bit ) ;
  depthEnum.insert ( "Ten_bit", Pds::Pulnix::TM6740ConfigV2::Ten_bit ) ;

  hdf5pp::EnumType<uint8_t> binningEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  binningEnum.insert ( "x1", Pds::Pulnix::TM6740ConfigV2::x1 ) ;
  binningEnum.insert ( "x2", Pds::Pulnix::TM6740ConfigV2::x2 ) ;
  binningEnum.insert ( "x4", Pds::Pulnix::TM6740ConfigV2::x4 ) ;

  hdf5pp::EnumType<uint8_t> lookupTableEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  lookupTableEnum.insert ( "Gamma", Pds::Pulnix::TM6740ConfigV2::Gamma ) ;
  lookupTableEnum.insert ( "Linear", Pds::Pulnix::TM6740ConfigV2::Linear ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<PulnixTM6740ConfigV2>() ;
  confType.insert_native<uint16_t>( "vref_a", offsetof(PulnixTM6740ConfigV2_Data,vref_a) ) ;
  confType.insert_native<uint16_t>( "vref_b", offsetof(PulnixTM6740ConfigV2_Data,vref_b) ) ;
  confType.insert_native<uint16_t>( "gain_a", offsetof(PulnixTM6740ConfigV2_Data,gain_a) ) ;
  confType.insert_native<uint16_t>( "gain_b", offsetof(PulnixTM6740ConfigV2_Data,gain_b) ) ;
  confType.insert_native<uint8_t>( "gain_balance", offsetof(PulnixTM6740ConfigV2_Data,gain_balance) ) ;
  confType.insert( "output_resolution", offsetof(PulnixTM6740ConfigV2_Data,output_resolution), depthEnum ) ;
  confType.insert_native<uint8_t>( "output_resolution_bits", offsetof(PulnixTM6740ConfigV2_Data,output_resolution_bits) ) ;
  confType.insert( "horizontal_binning", offsetof(PulnixTM6740ConfigV2_Data,horizontal_binning), binningEnum ) ;
  confType.insert( "vertical_binning", offsetof(PulnixTM6740ConfigV2_Data,vertical_binning), binningEnum ) ;
  confType.insert( "lookuptable_mode", offsetof(PulnixTM6740ConfigV2_Data,lookuptable_mode), lookupTableEnum ) ;

  return confType ;
}

void
PulnixTM6740ConfigV2::store( const Pds::Pulnix::TM6740ConfigV2& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  PulnixTM6740ConfigV2 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}


} // namespace H5DataTypes
