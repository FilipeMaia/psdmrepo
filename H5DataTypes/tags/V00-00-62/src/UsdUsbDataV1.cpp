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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
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
UsdUsbDataV1::UsdUsbDataV1 ( const XtcType& data )
  : timestamp(data.timestamp())
  , digital_in(data.digital_in())
{
  for (int i = 0; i != Encoder_Inputs; ++ i) {
    encoder_count[i] = data.encoder_count(i);
  }
  for (int i = 0; i != Analog_Inputs; ++ i) {
    analog_in[i] = data.analog_in(i);
  }
}

hdf5pp::Type
UsdUsbDataV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
UsdUsbDataV1::native_type()
{
  hdf5pp::ArrayType encoderCountType = hdf5pp::ArrayType::arrayType<uint32_t>(Encoder_Inputs);
  hdf5pp::ArrayType analogInType = hdf5pp::ArrayType::arrayType<uint16_t>(Analog_Inputs);

  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<UsdUsbDataV1>() ;
  type.insert( "encoder_count", offsetof(UsdUsbDataV1, encoder_count), encoderCountType ) ;
  type.insert( "analog_in", offsetof(UsdUsbDataV1, analog_in), analogInType ) ;
  type.insert_native<uint32_t>( "timestamp", offsetof(UsdUsbDataV1, timestamp) ) ;
  type.insert_native<uint8_t>( "digital_in", offsetof(UsdUsbDataV1, digital_in) ) ;

  return type;
}

} // namespace H5DataTypes
