//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OrcaConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/OrcaConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

OrcaConfigV1::OrcaConfigV1 ( const Pds::Orca::ConfigV1& data )
  : rows(data.rows())
  , mode(data.mode())
  , cooling(data.cooling())
  , defect_pixel_correction_enabled(data.defect_pixel_correction_enabled())
{
}

hdf5pp::Type
OrcaConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
OrcaConfigV1::native_type()
{
  hdf5pp::EnumType<uint8_t> readoutModeEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  readoutModeEnum.insert ( "x1", Pds::Orca::ConfigV1::x1 ) ;
  readoutModeEnum.insert ( "x2", Pds::Orca::ConfigV1::x2 ) ;
  readoutModeEnum.insert ( "x4", Pds::Orca::ConfigV1::x4 ) ;
  readoutModeEnum.insert ( "Subarray", Pds::Orca::ConfigV1::Subarray ) ;

  hdf5pp::EnumType<uint8_t> coolingEnum = hdf5pp::EnumType<uint8_t>::enumType() ;
  coolingEnum.insert ( "Off", Pds::Orca::ConfigV1::Off ) ;
  coolingEnum.insert ( "On", Pds::Orca::ConfigV1::On ) ;
  coolingEnum.insert ( "Max", Pds::Orca::ConfigV1::Max ) ;

  hdf5pp::CompoundType confType = hdf5pp::CompoundType::compoundType<OrcaConfigV1>() ;
  confType.insert_native<uint32_t>( "rows", offsetof(OrcaConfigV1, rows) );
  confType.insert( "mode", offsetof(OrcaConfigV1, mode), readoutModeEnum );
  confType.insert( "cooling", offsetof(OrcaConfigV1, cooling), coolingEnum );
  confType.insert_native<uint8_t>( "defect_pixel_correction_enabled", offsetof(OrcaConfigV1, defect_pixel_correction_enabled) );

  return confType ;
}

void
OrcaConfigV1::store( const Pds::Orca::ConfigV1& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  OrcaConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
