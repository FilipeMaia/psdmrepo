//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPimV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/BldDataPimV1.h"

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

BldDataPimV1::BldDataPimV1 ( const XtcType& data )
  : camConfig(data.camConfig)
  , pimConfig(data.pimConfig)
  , frame(data.frame)
{
}

BldDataPimV1::~BldDataPimV1()
{
}

hdf5pp::Type
BldDataPimV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
BldDataPimV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<BldDataPimV1>() ;

  type.insert( "camConfig", offsetof(BldDataPimV1,camConfig), PulnixTM6740ConfigV2::native_type() );
  type.insert( "pimConfig", offsetof(BldDataPimV1,pimConfig), LusiPimImageConfigV1::native_type() );
  type.insert( "frame", offsetof(BldDataPimV1,frame), CameraFrameV1::native_type() );

  return type ;
}

} // namespace H5DataTypes
