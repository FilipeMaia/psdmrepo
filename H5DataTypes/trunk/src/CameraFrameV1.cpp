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
#include "Lusi/Lusi.h"

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
CameraFrameV1::CameraFrameV1 ( const Pds::Camera::FrameV1& frame )
{
  m_data.xyz = 1 ;
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
  type.insert_native<double>( "xyz", offsetof(CameraFrameV1_Data,xyz) ) ;

  return type ;
}

} // namespace H5DataTypes
