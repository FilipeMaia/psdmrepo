//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixSamplerElementV1...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/EpixSamplerElementV1.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/ArrayType.h"
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

EpixSamplerElementV1::EpixSamplerElementV1 ( const XtcType& data )
  : vc(data.vc())
  , lane(data.lane())
  , acqCount(data.acqCount())
  , frameNumber(data.frameNumber())
  , ticks(data.ticks())
  , fiducials(data.fiducials())
{
}

hdf5pp::Type
EpixSamplerElementV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
EpixSamplerElementV1::native_type()
{
  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<EpixSamplerElementV1>() ;
  type.insert_native<uint8_t>( "vc", offsetof(EpixSamplerElementV1, vc) );
  type.insert_native<uint8_t>( "lane", offsetof(EpixSamplerElementV1, lane) );
  type.insert_native<uint16_t>( "acqCount", offsetof(EpixSamplerElementV1, acqCount) );
  type.insert_native<uint32_t>( "frameNumber", offsetof(EpixSamplerElementV1, frameNumber) );
  type.insert_native<uint32_t>( "ticks", offsetof(EpixSamplerElementV1, ticks) );
  type.insert_native<uint32_t>( "fiducials", offsetof(EpixSamplerElementV1, fiducials) );

  return type;
}

hdf5pp::Type
EpixSamplerElementV1::frame_data_type(int nChan, int samplesPerChannel)
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  hsize_t dims[] = { nChan, samplesPerChannel } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 2, dims );
}

hdf5pp::Type
EpixSamplerElementV1::temperature_data_type(int nChan)
{
  return hdf5pp::ArrayType::arrayType<uint16_t>(nChan);
}

} // namespace H5DataTypes
