//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraFrameV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CameraFrameV1.h"

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
CameraFrameV1::CameraFrameV1 ( const Pds::Camera::FrameV1& data )
  : width(data.width())
  , height(data.height())
  , depth(data.depth())
  , offset(data.offset())
{
}

CameraFrameV1::~CameraFrameV1 ()
{
}

hdf5pp::Type
CameraFrameV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CameraFrameV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<CameraFrameV1>() ;
  type.insert_native<uint32_t>( "width", offsetof(CameraFrameV1,width) ) ;
  type.insert_native<uint32_t>( "height", offsetof(CameraFrameV1,height) ) ;
  type.insert_native<uint32_t>( "depth", offsetof(CameraFrameV1,depth) ) ;
  type.insert_native<uint32_t>( "offset", offsetof(CameraFrameV1,offset) ) ;

  return type ;
}

hdf5pp::Type
CameraFrameV1::imageType( const Pds::Camera::FrameV1& frame )
{
  hdf5pp::Type baseType ;
  if ( frame.depth_bytes() == 1 ) {
    baseType = hdf5pp::TypeTraits<uint8_t>::native_type() ;
  } else if ( frame.depth_bytes() == 2 ) {
    baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;
  }

  hsize_t dims[] = { frame.height(), frame.width() } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

} // namespace H5DataTypes
