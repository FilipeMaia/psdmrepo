//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimepixDataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/TimepixDataV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/ArrayType.h"
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
TimepixDataV1::TimepixDataV1(const XtcType& data)
  : timestamp(data.timestamp())
  , frameCounter(data.frameCounter())
  , lostRows(data.lostRows())
{
}

hdf5pp::Type
TimepixDataV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
TimepixDataV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<TimepixDataV1>() ;
  type.insert_native<uint32_t>("timestamp", offsetof(TimepixDataV1, timestamp));
  type.insert_native<uint16_t>("frameCounter", offsetof(TimepixDataV1, frameCounter));
  type.insert_native<uint16_t>("lostRows", offsetof(TimepixDataV1, lostRows));

  return type;
}

hdf5pp::Type
TimepixDataV1::stored_data_type(uint32_t height, uint32_t width)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  hsize_t dims[] = { height, width } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

} // namespace H5DataTypes
