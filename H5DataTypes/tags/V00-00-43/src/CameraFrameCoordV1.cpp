//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraFrameCoordV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/CameraFrameCoordV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/CompoundType.h"
#include "hdf5pp/TypeTraits.h"
#include "H5DataTypes/H5DataUtils.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace H5DataTypes {

CameraFrameCoordV1::CameraFrameCoordV1( const Pds::Camera::FrameCoord& coord )
{
  m_data.column = coord.column ;
  m_data.row = coord.row ;
}

hdf5pp::Type
CameraFrameCoordV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
CameraFrameCoordV1::native_type()
{
  // make the type
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<Pds::Camera::FrameCoord>() ;
  type.insert_native<uint16_t>( "column", offsetof(CameraFrameCoordV1_Data,column) ) ;
  type.insert_native<uint16_t>( "row", offsetof(CameraFrameCoordV1_Data,row) ) ;

  return type ;
}

void
storeCameraFrameCoordV1 ( hsize_t size, const Pds::Camera::FrameCoord* coord, hdf5pp::Group grp, const char* name )
{
  const hsize_t np = size ;
  CameraFrameCoordV1 coords[np] ;
  for ( hsize_t i = 0 ; i < np ; ++ i ) coords[i] = CameraFrameCoordV1(coord[i]) ;
  storeDataObjects ( np, coords, name, grp ) ;
}

} // namespace H5DataTypes
