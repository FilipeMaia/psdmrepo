//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsDataV1...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "H5DataTypes/OceanOpticsDataV1.h"

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
OceanOpticsDataV1::OceanOpticsDataV1(const XtcType& data)
  : frameCounter(const_cast<XtcType&>(data).frameCounter())
  , numDelayedFrames(const_cast<XtcType&>(data).numDelayedFrames())
  , numDiscardFrames(const_cast<XtcType&>(data).numDiscardFrames())
  , timeFrameStart(Pds::ClockTime(const_cast<XtcType&>(data).timeFrameStart()))
  , timeFrameFirstData(Pds::ClockTime(const_cast<XtcType&>(data).timeFrameFirstData()))
  , timeFrameEnd(Pds::ClockTime(const_cast<XtcType&>(data).timeFrameEnd()))
  , numSpectraInData(const_cast<XtcType&>(data).numSpectraInData())
  , numSpectraInQueue(const_cast<XtcType&>(data).numSpectraInQueue())
  , numSpectraUnused(const_cast<XtcType&>(data).numSpectraUnused())
  , durationOfFrame(const_cast<XtcType&>(data).durationOfFrame())
{
}

hdf5pp::Type
OceanOpticsDataV1::stored_type()
{
  return native_type() ;
}

hdf5pp::Type
OceanOpticsDataV1::native_type()
{
  hdf5pp::Type clockTimeType = XtcClockTime::native_type();

  hdf5pp::CompoundType type = hdf5pp::CompoundType::compoundType<OceanOpticsDataV1>() ;
  type.insert_native<uint64_t>("frameCounter", offsetof(OceanOpticsDataV1, frameCounter));
  type.insert_native<uint64_t>("numDelayedFrames", offsetof(OceanOpticsDataV1, numDelayedFrames));
  type.insert_native<uint64_t>("numDiscardFrames", offsetof(OceanOpticsDataV1, numDiscardFrames));
  type.insert("timeFrameStart", offsetof(OceanOpticsDataV1, timeFrameStart), clockTimeType);
  type.insert("timeFrameFirstData", offsetof(OceanOpticsDataV1, timeFrameFirstData), clockTimeType);
  type.insert("timeFrameEnd", offsetof(OceanOpticsDataV1, timeFrameEnd), clockTimeType);
  type.insert_native<int8_t>("numSpectraInData", offsetof(OceanOpticsDataV1, numSpectraInData));
  type.insert_native<int8_t>("numSpectraInQueue", offsetof(OceanOpticsDataV1, numSpectraInQueue));
  type.insert_native<int8_t>("numSpectraUnused", offsetof(OceanOpticsDataV1, numSpectraUnused));
  type.insert_native<double>("durationOfFrame", offsetof(OceanOpticsDataV1, durationOfFrame));

  return type;
}

hdf5pp::Type
OceanOpticsDataV1::stored_data_type()
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<uint16_t>::native_type() ;

  hsize_t dims[] = { Pds::OceanOptics::DataV1::iNumPixels } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 1, dims );
}

hdf5pp::Type
OceanOpticsDataV1::stored_corrected_data_type()
{
  hdf5pp::Type baseType = hdf5pp::TypeTraits<double>::native_type() ;

  hsize_t dims[] = { Pds::OceanOptics::DataV1::iNumPixels } ;
  return hdf5pp::ArrayType::arrayType ( baseType, 1, dims );
}

} // namespace H5DataTypes
