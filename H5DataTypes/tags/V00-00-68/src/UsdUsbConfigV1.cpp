//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class UsdUsbConfigV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/UsdUsbConfigV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/H5DataUtils.h"
#include "hdf5pp/ArrayType.h"
#include "hdf5pp/EnumType.h"
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"

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
UsdUsbConfigV1::UsdUsbConfigV1 ( const XtcType& data )
{
  for (int i = 0; i != NCHANNELS; ++ i) {
    counting_mode[i] = data.counting_mode(i);
    quadrature_mode[i] = data.quadrature_mode(i);
  }
}

hdf5pp::Type
UsdUsbConfigV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
UsdUsbConfigV1::native_type()
{
  hdf5pp::EnumType<uint32_t> countModeEnumType = hdf5pp::EnumType<uint32_t>::enumType() ;
  countModeEnumType.insert ( "WRAP_FULL", Pds::UsdUsb::ConfigV1::WRAP_FULL ) ;
  countModeEnumType.insert ( "LIMIT", Pds::UsdUsb::ConfigV1::LIMIT ) ;
  countModeEnumType.insert ( "HALT", Pds::UsdUsb::ConfigV1::HALT ) ;
  countModeEnumType.insert ( "WRAP_PRESET", Pds::UsdUsb::ConfigV1::WRAP_PRESET ) ;

  hdf5pp::EnumType<uint32_t> quadModeEnumType = hdf5pp::EnumType<uint32_t>::enumType() ;
  quadModeEnumType.insert ( "CLOCK_DIR", Pds::UsdUsb::ConfigV1::CLOCK_DIR ) ;
  quadModeEnumType.insert ( "X1", Pds::UsdUsb::ConfigV1::X1 ) ;
  quadModeEnumType.insert ( "X2", Pds::UsdUsb::ConfigV1::X2 ) ;
  quadModeEnumType.insert ( "X4", Pds::UsdUsb::ConfigV1::X4 ) ;

  hdf5pp::ArrayType countModeType = hdf5pp::ArrayType::arrayType(countModeEnumType, NCHANNELS);
  hdf5pp::ArrayType quadModeType = hdf5pp::ArrayType::arrayType(quadModeEnumType, NCHANNELS);

  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<UsdUsbConfigV1>() ;
  type.insert( "counting_mode", offsetof(UsdUsbConfigV1, counting_mode), countModeType ) ;
  type.insert( "quadrature_mode", offsetof(UsdUsbConfigV1, quadrature_mode), quadModeType ) ;

  return type;
}

void
UsdUsbConfigV1::store( const XtcType& config, hdf5pp::Group grp )
{
  // make scalar data set for main object
  UsdUsbConfigV1 data ( config ) ;
  storeDataObject ( data, "config", grp ) ;
}

} // namespace H5DataTypes
