//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsDataV2...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/OceanOpticsDataV2.h"

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
OceanOpticsDataV2::OceanOpticsDataV2(const XtcType& data)
  : frameCounter(data.frameCounter())
  , numDelayedFrames(data.numDelayedFrames())
  , numDiscardFrames(data.numDiscardFrames())
  , timeFrameStart(data.timeFrameStart().tv_sec(), data.timeFrameStart().tv_nsec())
  , timeFrameFirstData(data.timeFrameFirstData().tv_sec(), data.timeFrameFirstData().tv_nsec())
  , timeFrameEnd(data.timeFrameEnd().tv_sec(), data.timeFrameEnd().tv_nsec())
  , numSpectraInData(data.numSpectraInData())
  , numSpectraInQueue(data.numSpectraInQueue())
  , numSpectraUnused(data.numSpectraUnused())
  , durationOfFrame(data.durationOfFrame())
{
}

hdf5pp::Type
OceanOpticsDataV2::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
OceanOpticsDataV2::native_type()
{
  hdf5pp::Type clockTimeType = XtcClockTime::native_type();

  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<OceanOpticsDataV2>() ;
  type.insert_native<uint64_t>("frameCounter", offsetof(OceanOpticsDataV2, frameCounter));
  type.insert_native<uint64_t>("numDelayedFrames", offsetof(OceanOpticsDataV2, numDelayedFrames));
  type.insert_native<uint64_t>("numDiscardFrames", offsetof(OceanOpticsDataV2, numDiscardFrames));
  type.insert("timeFrameStart", offsetof(OceanOpticsDataV2, timeFrameStart), clockTimeType);
  type.insert("timeFrameFirstData", offsetof(OceanOpticsDataV2, timeFrameFirstData), clockTimeType);
  type.insert("timeFrameEnd", offsetof(OceanOpticsDataV2, timeFrameEnd), clockTimeType);
  type.insert_native<int8_t>("numSpectraInData", offsetof(OceanOpticsDataV2, numSpectraInData));
  type.insert_native<int8_t>("numSpectraInQueue", offsetof(OceanOpticsDataV2, numSpectraInQueue));
  type.insert_native<int8_t>("numSpectraUnused", offsetof(OceanOpticsDataV2, numSpectraUnused));
  type.insert_native<double>("durationOfFrame", offsetof(OceanOpticsDataV2, durationOfFrame));

  return type;
}

hdf5pp::Type
OceanOpticsDataV2::stored_data_type()
{
  return hdf5pp::TypeTraits<uint16_t>::native_type(Pds::OceanOptics::DataV2::iNumPixels) ;
}

hdf5pp::Type
OceanOpticsDataV2::stored_corrected_data_type()
{
  return hdf5pp::TypeTraits<float>::native_type(Pds::OceanOptics::DataV2::iNumPixels) ;
}

} // namespace H5DataTypes
