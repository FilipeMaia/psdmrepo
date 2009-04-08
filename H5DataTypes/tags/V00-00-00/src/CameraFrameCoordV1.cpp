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
#include "Lusi/Lusi.h"

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
CameraFrameCoordV1::persType()
{
  // make the type
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<Pds::Camera::FrameCoord>() ;
  type.insert( "column", offsetof(CameraFrameCoordV1_Data,column), hdf5pp::AtomicType::atomicType<uint16_t>() ) ;
  type.insert( "row", offsetof(CameraFrameCoordV1_Data,row), hdf5pp::AtomicType::atomicType<uint16_t>() ) ;

  return type ;
}

void
storeCameraFrameCoordV1 ( hsize_t size, const Pds::Camera::FrameCoord* coord, hdf5pp::Group grp, const char* name )
{
  storeDataObjects ( size, coord, CameraFrameCoordV1::persType(), name, grp ) ;
}

} // namespace H5DataTypes
