//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonFrameV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/PrincetonFrameV2.h"

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
PrincetonFrameV2::PrincetonFrameV2 ( const XtcType& frame )
  : shotIdStart(frame.shotIdStart())
  , readoutTime(frame.readoutTime())
  , temperature(frame.temperature())
{
}

hdf5pp::Type
PrincetonFrameV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
PrincetonFrameV2::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<PrincetonFrameV2>() ;
  type.insert_native<uint32_t>( "shotIdStart", offsetof(PrincetonFrameV2, shotIdStart) ) ;
  type.insert_native<float>( "readoutTime", offsetof(PrincetonFrameV2, readoutTime) ) ;
  type.insert_native<float>( "temperature", offsetof(PrincetonFrameV2, temperature) ) ;

  return type;
}

hdf5pp::Type
PrincetonFrameV2::stored_data_type(uint32_t height, uint32_t width)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  hsize_t dims[] = { height, width } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

} // namespace H5DataTypes
