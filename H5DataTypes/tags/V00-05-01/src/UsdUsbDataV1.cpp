//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class UsdUsbDataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/UsdUsbDataV1.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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
UsdUsbDataV1::UsdUsbDataV1 ( const XtcType& data )
  : timestamp(data.timestamp())
  , digital_in(data.digital_in())
{
  const ndarray<const int32_t, 1>& nd_encoder_count = data.encoder_count();
  std::copy(nd_encoder_count.begin(), nd_encoder_count.end(), encoder_count);

  const ndarray<const uint16_t, 1>& nd_analog_in = data.analog_in();
  std::copy(nd_analog_in.begin(), nd_analog_in.end(), analog_in);

  const ndarray<const uint8_t, 1>& ndstatus = data.status();
  std::copy(ndstatus.begin(), ndstatus.end(), status);
}

hdf5pp::Type
UsdUsbDataV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
UsdUsbDataV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<UsdUsbDataV1>() ;
  type.insert_native<int32_t>( "encoder_count", offsetof(UsdUsbDataV1, encoder_count), Encoder_Inputs ) ;
  type.insert_native<uint16_t>( "analog_in", offsetof(UsdUsbDataV1, analog_in), Analog_Inputs ) ;
  type.insert_native<uint32_t>( "timestamp", offsetof(UsdUsbDataV1, timestamp) ) ;
  type.insert_native<uint8_t>( "status", offsetof(UsdUsbDataV1, status), 4 ) ;
  type.insert_native<uint8_t>( "digital_in", offsetof(UsdUsbDataV1, digital_in) ) ;

  return type;
}

} // namespace H5DataTypes
